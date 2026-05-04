#include "AeroPedNavSemanticSubsystem.h"

#include "Algo/Reverse.h"
#include "Dom/JsonObject.h"
#include "GameFramework/Actor.h"
#include "Misc/FileHelper.h"
#include "Misc/SecureHash.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
constexpr double MaxSemanticGroundZMismatchM = 0.5;

bool LoadJsonObjectFromFile(const FString& FilePath, TSharedPtr<FJsonObject>& OutObject, FString& OutError)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		OutError = FString::Printf(TEXT("Failed to read JSON file: %s"), *FilePath);
		return false;
	}

	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, OutObject) || !OutObject.IsValid())
	{
		OutError = FString::Printf(TEXT("Failed to parse JSON file: %s"), *FilePath);
		return false;
	}

	return true;
}

bool SaveJsonObjectToFile(const FString& FilePath, const TSharedPtr<FJsonObject>& RootObject, FString& OutError)
{
	FString Output;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	if (!FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer))
	{
		OutError = FString::Printf(TEXT("Failed to serialize JSON for %s"), *FilePath);
		return false;
	}

	if (!FFileHelper::SaveStringToFile(Output, *FilePath))
	{
		OutError = FString::Printf(TEXT("Failed to save JSON file: %s"), *FilePath);
		return false;
	}

	return true;
}

bool ReadVectorArray(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector)
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Array>(FieldName))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>& Values = Object->GetArrayField(FieldName);
	if (Values.Num() < 3)
	{
		return false;
	}

	OutVector.X = Values[0]->AsNumber();
	OutVector.Y = Values[1]->AsNumber();
	OutVector.Z = Values[2]->AsNumber();
	return true;
}

TArray<FVector> ReadPolyline(const TSharedPtr<FJsonObject>& Object, const FString& FieldName)
{
	TArray<FVector> Points;
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Array>(FieldName))
	{
		return Points;
	}

	const TArray<TSharedPtr<FJsonValue>>& PointValues = Object->GetArrayField(FieldName);
	for (const TSharedPtr<FJsonValue>& PointValue : PointValues)
	{
		const TArray<TSharedPtr<FJsonValue>>* Coords = nullptr;
		if (PointValue.IsValid() && PointValue->TryGetArray(Coords) && Coords != nullptr && Coords->Num() >= 3)
		{
			Points.Add(FVector((*Coords)[0]->AsNumber(), (*Coords)[1]->AsNumber(), (*Coords)[2]->AsNumber()));
		}
	}
	return Points;
}

void AppendVector(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& Value)
{
	TArray<TSharedPtr<FJsonValue>> Values;
	Values.Add(MakeShared<FJsonValueNumber>(Value.X));
	Values.Add(MakeShared<FJsonValueNumber>(Value.Y));
	Values.Add(MakeShared<FJsonValueNumber>(Value.Z));
	Object->SetArrayField(FieldName, Values);
}

void AppendPolyline(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const TArray<FVector>& Polyline)
{
	TArray<TSharedPtr<FJsonValue>> Values;
	for (const FVector& Point : Polyline)
	{
		TArray<TSharedPtr<FJsonValue>> PointValues;
		PointValues.Add(MakeShared<FJsonValueNumber>(Point.X));
		PointValues.Add(MakeShared<FJsonValueNumber>(Point.Y));
		PointValues.Add(MakeShared<FJsonValueNumber>(Point.Z));
		Values.Add(MakeShared<FJsonValueArray>(PointValues));
	}
	Object->SetArrayField(FieldName, Values);
}

double ComputePolylineLength(const TArray<FVector>& Polyline)
{
	double Length = 0.0;
	for (int32 Index = 1; Index < Polyline.Num(); ++Index)
	{
		Length += FVector::Distance(Polyline[Index - 1], Polyline[Index]);
	}
	return Length;
}

bool StringContainsAnyKeyword(const FString& Source, std::initializer_list<const TCHAR*> Keywords)
{
	for (const TCHAR* Keyword : Keywords)
	{
		if (Source.Contains(Keyword, ESearchCase::IgnoreCase))
		{
			return true;
		}
	}
	return false;
}

bool IsSemanticPreferredGroundActor(const AActor* Actor)
{
	if (!IsValid(Actor))
	{
		return false;
	}

	static const FName PreferredGroundTags[] = {
		FName(TEXT("terrain")),
		FName(TEXT("ground")),
		FName(TEXT("road")),
		FName(TEXT("sidewalk")),
		FName(TEXT("landscape")),
		FName(TEXT("bridge"))};
	for (const FName& Tag : PreferredGroundTags)
	{
		if (Actor->ActorHasTag(Tag))
		{
			return true;
		}
	}

	const FString ActorName = Actor->GetName();
	if (StringContainsAnyKeyword(ActorName, {TEXT("road"), TEXT("sidewalk"), TEXT("terrain"), TEXT("landscape"), TEXT("bridge"), TEXT("citybase")}))
	{
		return true;
	}

	const UClass* ActorClass = Actor->GetClass();
	return ActorClass != nullptr && StringContainsAnyKeyword(ActorClass->GetName(), {TEXT("Road"), TEXT("Landscape"), TEXT("Bridge"), TEXT("CityBase")});
}

bool SampleGroundPointFromWorld(
	const UWorld* World,
	const FVector& WorldOriginCm,
	const FVector& InputEnuM,
	double TraceHalfHeightM,
	FVector& OutProjectedEnuM,
	FVector& OutSurfaceNormalEnu)
{
	if (World == nullptr)
	{
		return false;
	}

	const FVector TraceOriginCm = WorldOriginCm + InputEnuM * 100.0;
	auto TryTrace = [&](const double EffectiveTraceHalfHeightM) -> bool
	{
		const FVector TraceStart = TraceOriginCm + FVector(0.0, 0.0, EffectiveTraceHalfHeightM * 100.0);
		const FVector TraceEnd = TraceOriginCm - FVector(0.0, 0.0, EffectiveTraceHalfHeightM * 100.0);
		TArray<FHitResult> HitResults;
		FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(AeroPedSemanticCompileGroundProjection), false);
		if (!World->LineTraceMultiByChannel(HitResults, TraceStart, TraceEnd, ECollisionChannel::ECC_Visibility, QueryParams))
		{
			return false;
		}

		const FHitResult* SelectedHit = nullptr;
		for (const FHitResult& HitResult : HitResults)
		{
			if (!HitResult.bBlockingHit)
			{
				continue;
			}

			if (SelectedHit == nullptr || HitResult.ImpactPoint.Z > SelectedHit->ImpactPoint.Z)
			{
				SelectedHit = &HitResult;
			}

			if (IsSemanticPreferredGroundActor(HitResult.GetActor()))
			{
				SelectedHit = &HitResult;
				break;
			}
		}

		if (SelectedHit == nullptr)
		{
			return false;
		}

		OutProjectedEnuM = (SelectedHit->ImpactPoint - WorldOriginCm) / 100.0;
		OutSurfaceNormalEnu = FVector(SelectedHit->ImpactNormal).GetSafeNormal();
		return true;
	};

	if (TryTrace(TraceHalfHeightM))
	{
		return true;
	}

	return TraceHalfHeightM >= 500.0 ? false : TryTrace(500.0);
}

void SamplePolylineAgainstWorld(
	const UWorld* World,
	const FVector& WorldOriginCm,
	double TraceHalfHeightM,
	TArray<FVector>& InOutPolylineEnuM,
	TArray<FVector>& OutNormalsEnu)
{
	OutNormalsEnu.Reset();
	OutNormalsEnu.Reserve(InOutPolylineEnuM.Num());
	for (FVector& PointEnuM : InOutPolylineEnuM)
	{
		FVector SampledPointEnuM = PointEnuM;
		FVector SurfaceNormalEnu = FVector::UpVector;
		if (SampleGroundPointFromWorld(World, WorldOriginCm, PointEnuM, TraceHalfHeightM, SampledPointEnuM, SurfaceNormalEnu))
		{
			PointEnuM = SampledPointEnuM;
		}
		OutNormalsEnu.Add(SurfaceNormalEnu);
	}
}

void AppendVectorList(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const TArray<FVector>& ValuesIn)
{
	TArray<TSharedPtr<FJsonValue>> Values;
	for (const FVector& Value : ValuesIn)
	{
		TArray<TSharedPtr<FJsonValue>> PointValues;
		PointValues.Add(MakeShared<FJsonValueNumber>(Value.X));
		PointValues.Add(MakeShared<FJsonValueNumber>(Value.Y));
		PointValues.Add(MakeShared<FJsonValueNumber>(Value.Z));
		Values.Add(MakeShared<FJsonValueArray>(PointValues));
	}
	Object->SetArrayField(FieldName, Values);
}
}

bool UAeroPedNavSemanticSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroPedNavSemanticSubsystem::SetMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext)
{
	CurrentMapId = MapId;
	CurrentWorldOriginCm = FVector::ZeroVector;
	if (MapContext.IsValid())
	{
		TryReadVectorField(MapContext, TEXT("world_origin_cm"), CurrentWorldOriginCm);
	}
}

bool UAeroPedNavSemanticSubsystem::LoadSemanticSource(const FString& SourcePath, FString& OutError)
{
	return LoadJsonObjectFromFile(SourcePath, SourceDocument, OutError);
}

bool UAeroPedNavSemanticSubsystem::LoadSemanticBundle(const FString& BundlePath, FString& OutError)
{
	if (!LoadJsonObjectFromFile(BundlePath, BundleDocument, OutError))
	{
		return false;
	}
	return RebuildRuntimeCacheFromBundle(BundleDocument, OutError);
}

bool UAeroPedNavSemanticSubsystem::CompileSemanticBundle(const FString& SourcePath, const FString& BundlePath, FString& OutError)
{
	TSharedPtr<FJsonObject> SourceObject;
	if (!LoadJsonObjectFromFile(SourcePath, SourceObject, OutError))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* SidewalkSegments = nullptr;
	const TArray<TSharedPtr<FJsonValue>>* CrossingConnectors = nullptr;
	const TArray<TSharedPtr<FJsonValue>>* WaitingZones = nullptr;
	SourceObject->TryGetArrayField(TEXT("sidewalk_segments"), SidewalkSegments);
	SourceObject->TryGetArrayField(TEXT("crossing_connectors"), CrossingConnectors);
	SourceObject->TryGetArrayField(TEXT("waiting_zones"), WaitingZones);

	double BundleMaxSnapDistanceM = MaxSnapDistanceM;
	double BundleGroundTraceHalfHeightM = GroundTraceHalfHeightM;
	if (SourceObject->HasTypedField<EJson::Object>(TEXT("projection_rules")))
	{
		const TSharedPtr<FJsonObject> Rules = SourceObject->GetObjectField(TEXT("projection_rules"));
		if (Rules.IsValid())
		{
			Rules->TryGetNumberField(TEXT("max_snap_distance_m"), BundleMaxSnapDistanceM);
			Rules->TryGetNumberField(TEXT("ground_trace_half_height_m"), BundleGroundTraceHalfHeightM);
		}
	}

	TMap<FString, FVector> AnchorPositions;
	TMap<FString, FVector> AnchorSurfaceNormals;
	TMap<FString, TArray<FString>> AnchorWaitingZones;
	TArray<TSharedPtr<FJsonValue>> WalkEdges;
	TArray<TSharedPtr<FJsonValue>> Crosswalks;
	TArray<TSharedPtr<FJsonValue>> WaitPoints;
	const UWorld* SampleWorld = GetWorld();

	auto RegisterEdge = [SampleWorld, this, BundleGroundTraceHalfHeightM, &AnchorPositions, &AnchorSurfaceNormals](const TSharedPtr<FJsonObject>& EdgeObject, const FString& IdFieldName, const FString& TypeName, TArray<TSharedPtr<FJsonValue>>& TargetArray)
	{
		FString EdgeId;
		FString FromAnchorId;
		FString ToAnchorId;
		if (!EdgeObject.IsValid() ||
			!EdgeObject->TryGetStringField(IdFieldName, EdgeId) ||
			!EdgeObject->TryGetStringField(TEXT("from_anchor_id"), FromAnchorId) ||
			!EdgeObject->TryGetStringField(TEXT("to_anchor_id"), ToAnchorId))
		{
			return;
		}

		TArray<FVector> Polyline = ReadPolyline(EdgeObject, TEXT("polyline_enu_m"));
		if (Polyline.Num() < 2)
		{
			return;
		}

		TArray<FVector> PolylineNormalsEnu;
		SamplePolylineAgainstWorld(SampleWorld, CurrentWorldOriginCm, BundleGroundTraceHalfHeightM, Polyline, PolylineNormalsEnu);

		AnchorPositions.FindOrAdd(FromAnchorId) = Polyline[0];
		AnchorPositions.FindOrAdd(ToAnchorId) = Polyline.Last();
		AnchorSurfaceNormals.FindOrAdd(FromAnchorId) = PolylineNormalsEnu.Num() > 0 ? PolylineNormalsEnu[0] : FVector::UpVector;
		AnchorSurfaceNormals.FindOrAdd(ToAnchorId) = PolylineNormalsEnu.Num() > 0 ? PolylineNormalsEnu.Last() : FVector::UpVector;

		TSharedPtr<FJsonObject> RuntimeEdge = MakeShared<FJsonObject>();
		RuntimeEdge->SetStringField(TEXT("edge_id"), EdgeId);
		RuntimeEdge->SetStringField(TEXT("edge_type"), TypeName);
		RuntimeEdge->SetStringField(TEXT("from_anchor_id"), FromAnchorId);
		RuntimeEdge->SetStringField(TEXT("to_anchor_id"), ToAnchorId);
		RuntimeEdge->SetNumberField(TEXT("length_m"), ComputePolylineLength(Polyline));
		AppendPolyline(RuntimeEdge, TEXT("polyline_enu_m"), Polyline);
		AppendVectorList(RuntimeEdge, TEXT("polyline_normals_enu"), PolylineNormalsEnu);
		TargetArray.Add(MakeShared<FJsonValueObject>(RuntimeEdge));
	};

	if (SidewalkSegments != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *SidewalkSegments)
		{
			RegisterEdge(Value->AsObject(), TEXT("segment_id"), TEXT("sidewalk"), WalkEdges);
		}
	}

	if (CrossingConnectors != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *CrossingConnectors)
		{
			RegisterEdge(Value->AsObject(), TEXT("connector_id"), TEXT("crosswalk"), Crosswalks);
		}
	}

	if (WaitingZones != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *WaitingZones)
		{
			const TSharedPtr<FJsonObject> ZoneObject = Value->AsObject();
			FString ZoneId;
			FString AnchorId;
			FVector Center = FVector::ZeroVector;
			if (!ZoneObject.IsValid() ||
				!ZoneObject->TryGetStringField(TEXT("zone_id"), ZoneId) ||
				!ZoneObject->TryGetStringField(TEXT("anchor_id"), AnchorId) ||
				!ReadVectorArray(ZoneObject, TEXT("center_enu_m"), Center))
			{
				continue;
			}

			FVector SurfaceNormalEnu = FVector::UpVector;
			FVector SampledCenterEnuM = Center;
			if (SampleGroundPointFromWorld(SampleWorld, CurrentWorldOriginCm, Center, BundleGroundTraceHalfHeightM, SampledCenterEnuM, SurfaceNormalEnu))
			{
				Center = SampledCenterEnuM;
			}

			AnchorPositions.FindOrAdd(AnchorId) = Center;
			AnchorSurfaceNormals.FindOrAdd(AnchorId) = SurfaceNormalEnu;
			AnchorWaitingZones.FindOrAdd(AnchorId).Add(ZoneId);

			TSharedPtr<FJsonObject> WaitPoint = MakeShared<FJsonObject>();
			WaitPoint->SetStringField(TEXT("zone_id"), ZoneId);
			WaitPoint->SetStringField(TEXT("anchor_id"), AnchorId);
			WaitPoint->SetNumberField(TEXT("radius_m"), ZoneObject->HasField(TEXT("radius_m")) ? ZoneObject->GetNumberField(TEXT("radius_m")) : 2.0);
			AppendVector(WaitPoint, TEXT("center_enu_m"), Center);
			AppendVector(WaitPoint, TEXT("surface_normal_enu"), SurfaceNormalEnu);
			WaitPoints.Add(MakeShared<FJsonValueObject>(WaitPoint));
		}
	}

	TArray<TSharedPtr<FJsonValue>> AnchorsArray;
	for (const TPair<FString, FVector>& Pair : AnchorPositions)
	{
		TSharedPtr<FJsonObject> AnchorObject = MakeShared<FJsonObject>();
		AnchorObject->SetStringField(TEXT("anchor_id"), Pair.Key);
		AppendVector(AnchorObject, TEXT("position_enu_m"), Pair.Value);
		AppendVector(AnchorObject, TEXT("surface_normal_enu"), AnchorSurfaceNormals.FindRef(Pair.Key).IsNearlyZero() ? FVector::UpVector : AnchorSurfaceNormals.FindRef(Pair.Key));

		TArray<TSharedPtr<FJsonValue>> WaitingZoneIds;
		for (const FString& ZoneId : AnchorWaitingZones.FindRef(Pair.Key))
		{
			WaitingZoneIds.Add(MakeShared<FJsonValueString>(ZoneId));
		}
		AnchorObject->SetArrayField(TEXT("waiting_zone_ids"), WaitingZoneIds);
		AnchorsArray.Add(MakeShared<FJsonValueObject>(AnchorObject));
	}

	TSharedPtr<FJsonObject> BundleObject = MakeShared<FJsonObject>();
	BundleObject->SetArrayField(TEXT("anchors"), AnchorsArray);
	BundleObject->SetArrayField(TEXT("walk_edges"), WalkEdges);
	BundleObject->SetArrayField(TEXT("wait_points"), WaitPoints);
	BundleObject->SetArrayField(TEXT("crosswalks"), Crosswalks);
	BundleObject->SetArrayField(TEXT("renderables"), TArray<TSharedPtr<FJsonValue>>());
	BundleObject->SetNumberField(TEXT("max_snap_distance_m"), BundleMaxSnapDistanceM);
	BundleObject->SetNumberField(TEXT("ground_trace_half_height_m"), BundleGroundTraceHalfHeightM);
	MaxSnapDistanceM = BundleMaxSnapDistanceM;
	GroundTraceHalfHeightM = BundleGroundTraceHalfHeightM;

	FString SourceContent;
	FFileHelper::LoadFileToString(SourceContent, *SourcePath);
	BundleObject->SetStringField(TEXT("source_hash"), FMD5::HashAnsiString(*SourceContent));

	if (!SaveJsonObjectToFile(BundlePath, BundleObject, OutError))
	{
		return false;
	}

	SourceDocument = SourceObject;
	BundleDocument = BundleObject;
	return RebuildRuntimeCacheFromBundle(BundleDocument, OutError);
}

TSharedPtr<FJsonObject> UAeroPedNavSemanticSubsystem::QueryPedPath(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("QueryPedPath payload is invalid.");
		return nullptr;
	}

	FVector OriginEnuM = FVector::ZeroVector;
	FVector DestinationEnuM = FVector::ZeroVector;
	if (!TryReadVectorField(Payload, TEXT("origin_enu_m"), OriginEnuM) || !TryReadVectorField(Payload, TEXT("destination_enu_m"), DestinationEnuM))
	{
		OutError = TEXT("QueryPedPath requires origin_enu_m and destination_enu_m.");
		return nullptr;
	}

	FVector ProjectedOrigin = OriginEnuM;
	FVector ProjectedDestination = DestinationEnuM;
	FString OriginAnchorId;
	FString DestinationAnchorId;
	ProjectPointToGround(OriginEnuM, ProjectedOrigin, OriginAnchorId);
	ProjectPointToGround(DestinationEnuM, ProjectedDestination, DestinationAnchorId);

	TArray<FVector> Polyline;
	Polyline.Add(ProjectedOrigin);
	TArray<FString> AnchorPath;
	TArray<FString> EdgeIds;
	TArray<FString> WaitingZoneIds;
	TArray<FString> CrossingIds;

	if (!OriginAnchorId.IsEmpty() && !DestinationAnchorId.IsEmpty() && AnchorsById.Contains(OriginAnchorId) && AnchorsById.Contains(DestinationAnchorId))
	{
		TMap<FString, double> Distances;
		TMap<FString, FString> PreviousAnchor;
		TMap<FString, FAeroPedEdgeRuntime> PreviousEdge;
		TSet<FString> Visited;

		for (const TPair<FString, FAeroPedAnchorRuntime>& Pair : AnchorsById)
		{
			Distances.Add(Pair.Key, TNumericLimits<double>::Max());
		}
		Distances.FindOrAdd(OriginAnchorId) = 0.0;

		while (Visited.Num() < Distances.Num())
		{
			FString CurrentAnchorId;
			double CurrentDistance = TNumericLimits<double>::Max();
			for (const TPair<FString, double>& Pair : Distances)
			{
				if (!Visited.Contains(Pair.Key) && Pair.Value < CurrentDistance)
				{
					CurrentAnchorId = Pair.Key;
					CurrentDistance = Pair.Value;
				}
			}

			if (CurrentAnchorId.IsEmpty())
			{
				break;
			}

			Visited.Add(CurrentAnchorId);
			if (CurrentAnchorId == DestinationAnchorId)
			{
				break;
			}

			const TArray<FAeroPedEdgeRuntime>* OutgoingEdges = OutgoingEdgesByAnchorId.Find(CurrentAnchorId);
			if (OutgoingEdges == nullptr)
			{
				continue;
			}

			for (const FAeroPedEdgeRuntime& Edge : *OutgoingEdges)
			{
				const double CandidateDistance = CurrentDistance + Edge.LengthM;
				double& BestKnownDistance = Distances.FindOrAdd(Edge.ToAnchorId);
				if (CandidateDistance < BestKnownDistance)
				{
					BestKnownDistance = CandidateDistance;
					PreviousAnchor.Add(Edge.ToAnchorId, CurrentAnchorId);
					PreviousEdge.Add(Edge.ToAnchorId, Edge);
				}
			}
		}

		if (PreviousEdge.Contains(DestinationAnchorId) || OriginAnchorId == DestinationAnchorId)
		{
			TArray<FAeroPedEdgeRuntime> ReversedEdges;
			FString Cursor = DestinationAnchorId;
			AnchorPath.Add(Cursor);
			while (Cursor != OriginAnchorId)
			{
				const FAeroPedEdgeRuntime* Edge = PreviousEdge.Find(Cursor);
				const FString* Prev = PreviousAnchor.Find(Cursor);
				if (Edge == nullptr || Prev == nullptr)
				{
					break;
				}
				ReversedEdges.Add(*Edge);
				Cursor = *Prev;
				AnchorPath.Add(Cursor);
			}
			Algo::Reverse(ReversedEdges);
			Algo::Reverse(AnchorPath);

			for (const FAeroPedEdgeRuntime& Edge : ReversedEdges)
			{
				EdgeIds.Add(Edge.EdgeId);
				if (Edge.EdgeType.Equals(TEXT("crosswalk"), ESearchCase::IgnoreCase))
				{
					CrossingIds.Add(Edge.EdgeId);
				}

				for (const FVector& Point : Edge.PolylineEnuM)
				{
					if (Polyline.Num() == 0 || !Polyline.Last().Equals(Point, KINDA_SMALL_NUMBER))
					{
						Polyline.Add(Point);
					}
				}
			}

			for (const FString& AnchorId : AnchorPath)
			{
				if (const FAeroPedAnchorRuntime* Anchor = AnchorsById.Find(AnchorId))
				{
					WaitingZoneIds.Append(Anchor->WaitingZoneIds);
				}
			}
		}
	}

	if (Polyline.Num() == 1)
	{
		Polyline.Add(ProjectedDestination);
	}
	else if (!Polyline.Last().Equals(ProjectedDestination, KINDA_SMALL_NUMBER))
	{
		Polyline.Add(ProjectedDestination);
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	AppendVector(Result, TEXT("origin_projected_enu_m"), ProjectedOrigin);
	AppendVector(Result, TEXT("destination_projected_enu_m"), ProjectedDestination);
	AppendPolyline(Result, TEXT("polyline_enu_m"), Polyline);

	TArray<TSharedPtr<FJsonValue>> AnchorValues;
	for (const FString& AnchorId : AnchorPath)
	{
		AnchorValues.Add(MakeShared<FJsonValueString>(AnchorId));
	}
	Result->SetArrayField(TEXT("anchor_path"), AnchorValues);

	TArray<TSharedPtr<FJsonValue>> EdgeValues;
	for (const FString& EdgeId : EdgeIds)
	{
		EdgeValues.Add(MakeShared<FJsonValueString>(EdgeId));
	}
	Result->SetArrayField(TEXT("edge_ids"), EdgeValues);

	TArray<TSharedPtr<FJsonValue>> WaitingValues;
	for (const FString& ZoneId : WaitingZoneIds)
	{
		WaitingValues.Add(MakeShared<FJsonValueString>(ZoneId));
	}
	Result->SetArrayField(TEXT("waiting_zone_ids"), WaitingValues);

	TArray<TSharedPtr<FJsonValue>> CrossingValues;
	for (const FString& CrossingId : CrossingIds)
	{
		CrossingValues.Add(MakeShared<FJsonValueString>(CrossingId));
	}
	Result->SetArrayField(TEXT("crossing_connector_ids"), CrossingValues);
	return Result;
}

TSharedPtr<FJsonObject> UAeroPedNavSemanticSubsystem::ProjectGround(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("ProjectGround payload is invalid.");
		return nullptr;
	}

	FVector PointEnuM = FVector::ZeroVector;
	if (!TryReadVectorField(Payload, TEXT("point_enu_m"), PointEnuM))
	{
		OutError = TEXT("ProjectGround requires point_enu_m.");
		return nullptr;
	}

	FVector ProjectedEnuM = PointEnuM;
	FVector SurfaceNormalEnu = FVector::UpVector;
	FString AnchorId;
	ProjectPointToGroundDetailed(PointEnuM, ProjectedEnuM, SurfaceNormalEnu, AnchorId);

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	AppendVector(Result, TEXT("projected_enu_m"), ProjectedEnuM);
	AppendVector(Result, TEXT("surface_normal_enu"), SurfaceNormalEnu);
	Result->SetStringField(TEXT("anchor_id"), AnchorId);
	return Result;
}

TSharedPtr<FJsonObject> UAeroPedNavSemanticSubsystem::QueryPedAnchor(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("QueryPedAnchor payload is invalid.");
		return nullptr;
	}

	FString AnchorId;
	if (!Payload->TryGetStringField(TEXT("anchor_id"), AnchorId))
	{
		OutError = TEXT("QueryPedAnchor requires anchor_id.");
		return nullptr;
	}

	const FAeroPedAnchorRuntime* Anchor = AnchorsById.Find(AnchorId);
	if (Anchor == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown anchor_id: %s"), *AnchorId);
		return nullptr;
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("anchor_id"), Anchor->AnchorId);
	AppendVector(Result, TEXT("position_enu_m"), Anchor->PositionEnuM);
	AppendVector(Result, TEXT("surface_normal_enu"), Anchor->SurfaceNormalEnu);
	TArray<TSharedPtr<FJsonValue>> WaitingValues;
	for (const FString& ZoneId : Anchor->WaitingZoneIds)
	{
		WaitingValues.Add(MakeShared<FJsonValueString>(ZoneId));
	}
	Result->SetArrayField(TEXT("waiting_zone_ids"), WaitingValues);
	return Result;
}

bool UAeroPedNavSemanticSubsystem::RebuildRuntimeCacheFromBundle(const TSharedPtr<FJsonObject>& BundleObject, FString& OutError)
{
	if (!BundleObject.IsValid())
	{
		OutError = TEXT("Ped semantic bundle is invalid.");
		return false;
	}

	AnchorsById.Reset();
	OutgoingEdgesByAnchorId.Reset();
	WaitingZonesById.Reset();
	BundleDocument = BundleObject;

	BundleObject->TryGetNumberField(TEXT("max_snap_distance_m"), MaxSnapDistanceM);
	BundleObject->TryGetNumberField(TEXT("ground_trace_half_height_m"), GroundTraceHalfHeightM);

	const TArray<TSharedPtr<FJsonValue>>* Anchors = nullptr;
	if (BundleObject->TryGetArrayField(TEXT("anchors"), Anchors) && Anchors != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *Anchors)
		{
			const TSharedPtr<FJsonObject> AnchorObject = Value->AsObject();
			if (!AnchorObject.IsValid())
			{
				continue;
			}

			FAeroPedAnchorRuntime Anchor;
			AnchorObject->TryGetStringField(TEXT("anchor_id"), Anchor.AnchorId);
			ReadVectorArray(AnchorObject, TEXT("position_enu_m"), Anchor.PositionEnuM);
			ReadVectorArray(AnchorObject, TEXT("surface_normal_enu"), Anchor.SurfaceNormalEnu);
			const TArray<TSharedPtr<FJsonValue>>* WaitingZoneIds = nullptr;
			if (AnchorObject->TryGetArrayField(TEXT("waiting_zone_ids"), WaitingZoneIds) && WaitingZoneIds != nullptr)
			{
				for (const TSharedPtr<FJsonValue>& ZoneValue : *WaitingZoneIds)
				{
					Anchor.WaitingZoneIds.Add(ZoneValue->AsString());
				}
			}
			AnchorsById.Add(Anchor.AnchorId, Anchor);
		}
	}

	const TArray<TSharedPtr<FJsonValue>>* WaitPoints = nullptr;
	if (BundleObject->TryGetArrayField(TEXT("wait_points"), WaitPoints) && WaitPoints != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *WaitPoints)
		{
			const TSharedPtr<FJsonObject> ZoneObject = Value->AsObject();
			if (!ZoneObject.IsValid())
			{
				continue;
			}
			FString ZoneId;
			if (ZoneObject->TryGetStringField(TEXT("zone_id"), ZoneId))
			{
				WaitingZonesById.Add(ZoneId, ZoneObject);
			}
		}
	}

	auto RegisterEdgeArray = [this](const TArray<TSharedPtr<FJsonValue>>* Values)
	{
		if (Values == nullptr)
		{
			return;
		}

		for (const TSharedPtr<FJsonValue>& Value : *Values)
		{
			const TSharedPtr<FJsonObject> EdgeObject = Value->AsObject();
			if (!EdgeObject.IsValid())
			{
				continue;
			}

			FAeroPedEdgeRuntime Edge;
			EdgeObject->TryGetStringField(TEXT("edge_id"), Edge.EdgeId);
			EdgeObject->TryGetStringField(TEXT("edge_type"), Edge.EdgeType);
			EdgeObject->TryGetStringField(TEXT("from_anchor_id"), Edge.FromAnchorId);
			EdgeObject->TryGetStringField(TEXT("to_anchor_id"), Edge.ToAnchorId);
			EdgeObject->TryGetNumberField(TEXT("length_m"), Edge.LengthM);
			Edge.PolylineEnuM = ReadPolyline(EdgeObject, TEXT("polyline_enu_m"));
			Edge.PolylineNormalsEnu = ReadPolyline(EdgeObject, TEXT("polyline_normals_enu"));
			if (Edge.PolylineNormalsEnu.Num() != Edge.PolylineEnuM.Num())
			{
				Edge.PolylineNormalsEnu.Init(FVector::UpVector, Edge.PolylineEnuM.Num());
			}
			OutgoingEdgesByAnchorId.FindOrAdd(Edge.FromAnchorId).Add(Edge);

			FAeroPedEdgeRuntime ReverseEdge = Edge;
			Swap(ReverseEdge.FromAnchorId, ReverseEdge.ToAnchorId);
			Algo::Reverse(ReverseEdge.PolylineEnuM);
			Algo::Reverse(ReverseEdge.PolylineNormalsEnu);
			OutgoingEdgesByAnchorId.FindOrAdd(ReverseEdge.FromAnchorId).Add(ReverseEdge);
		}
	};

	const TArray<TSharedPtr<FJsonValue>>* WalkEdges = nullptr;
	BundleObject->TryGetArrayField(TEXT("walk_edges"), WalkEdges);
	RegisterEdgeArray(WalkEdges);

	const TArray<TSharedPtr<FJsonValue>>* Crosswalks = nullptr;
	BundleObject->TryGetArrayField(TEXT("crosswalks"), Crosswalks);
	RegisterEdgeArray(Crosswalks);
	return true;
}

FVector UAeroPedNavSemanticSubsystem::ConvertEnuMetersToWorldCm(const FVector& PositionEnuM) const
{
	return CurrentWorldOriginCm + PositionEnuM * 100.0;
}

bool UAeroPedNavSemanticSubsystem::TryReadVectorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector) const
{
	if (ReadVectorArray(Object, FieldName, OutVector))
	{
		return true;
	}

	if (Object.IsValid() && Object->HasTypedField<EJson::Object>(FieldName))
	{
		const TSharedPtr<FJsonObject> VectorObject = Object->GetObjectField(FieldName);
		if (VectorObject.IsValid())
		{
			const double X = VectorObject->HasField(TEXT("x")) ? VectorObject->GetNumberField(TEXT("x")) : VectorObject->GetNumberField(TEXT("east_m"));
			const double Y = VectorObject->HasField(TEXT("y")) ? VectorObject->GetNumberField(TEXT("y")) : VectorObject->GetNumberField(TEXT("north_m"));
			const double Z = VectorObject->HasField(TEXT("z")) ? VectorObject->GetNumberField(TEXT("z")) : VectorObject->GetNumberField(TEXT("up_m"));
			OutVector = FVector(X, Y, Z);
			return true;
		}
	}

	return false;
}

bool IsPreferredGroundActor(const AActor* Actor)
{
	return IsSemanticPreferredGroundActor(Actor);
}

bool TryProjectPointOntoSemanticEdges(
	const TMap<FString, TArray<FAeroPedEdgeRuntime>>& OutgoingEdgesByAnchorId,
	const TMap<FString, FAeroPedAnchorRuntime>& AnchorsById,
	const FVector& InputEnuM,
	const FVector& ReferenceEnuM,
	double MaxSnapDistanceM,
	FVector& OutProjectedEnuM,
	FVector& OutSurfaceNormalEnu,
	FString& OutAnchorId)
{
	bool bFoundProjection = false;
	double BestDistance2D = MaxSnapDistanceM;
	double BestZDelta = TNumericLimits<double>::Max();
	OutProjectedEnuM = InputEnuM;
	OutSurfaceNormalEnu = FVector::UpVector;
	OutAnchorId.Reset();

	for (const TPair<FString, TArray<FAeroPedEdgeRuntime>>& Pair : OutgoingEdgesByAnchorId)
	{
		for (const FAeroPedEdgeRuntime& Edge : Pair.Value)
		{
			if (Edge.PolylineEnuM.Num() < 2)
			{
				continue;
			}

			for (int32 Index = 1; Index < Edge.PolylineEnuM.Num(); ++Index)
			{
				const FVector& SegmentStart = Edge.PolylineEnuM[Index - 1];
				const FVector& SegmentEnd = Edge.PolylineEnuM[Index];
				const FVector2D SegmentDelta2D = FVector2D(SegmentEnd.X - SegmentStart.X, SegmentEnd.Y - SegmentStart.Y);
				const double SegmentLengthSquared2D = SegmentDelta2D.SizeSquared();
				if (SegmentLengthSquared2D <= KINDA_SMALL_NUMBER)
				{
					continue;
				}

				const FVector2D ToInput2D = FVector2D(InputEnuM.X - SegmentStart.X, InputEnuM.Y - SegmentStart.Y);
				const double SegmentT = FMath::Clamp(FVector2D::DotProduct(ToInput2D, SegmentDelta2D) / SegmentLengthSquared2D, 0.0, 1.0);
				const FVector CandidatePointEnuM = FMath::Lerp(SegmentStart, SegmentEnd, SegmentT);
				const double CandidateDistance2D = FVector2D::Distance(FVector2D(InputEnuM.X, InputEnuM.Y), FVector2D(CandidatePointEnuM.X, CandidatePointEnuM.Y));
				if (CandidateDistance2D > MaxSnapDistanceM)
				{
					continue;
				}

				const double CandidateZDelta = FMath::Abs(CandidatePointEnuM.Z - ReferenceEnuM.Z);
				const bool bIsBetterCandidate =
					!bFoundProjection ||
					CandidateDistance2D < BestDistance2D - KINDA_SMALL_NUMBER ||
					(FMath::IsNearlyEqual(CandidateDistance2D, BestDistance2D, 0.05) && CandidateZDelta < BestZDelta);
				if (!bIsBetterCandidate)
				{
					continue;
				}

				FVector CandidateNormalEnu = FVector::UpVector;
				if (Edge.PolylineNormalsEnu.IsValidIndex(Index - 1) && Edge.PolylineNormalsEnu.IsValidIndex(Index))
				{
					CandidateNormalEnu = FMath::Lerp(Edge.PolylineNormalsEnu[Index - 1], Edge.PolylineNormalsEnu[Index], SegmentT).GetSafeNormal();
					if (CandidateNormalEnu.IsNearlyZero())
					{
						CandidateNormalEnu = FVector::UpVector;
					}
				}

				bFoundProjection = true;
				BestDistance2D = CandidateDistance2D;
				BestZDelta = CandidateZDelta;
				OutProjectedEnuM = CandidatePointEnuM;
				OutSurfaceNormalEnu = CandidateNormalEnu;
			}
		}
	}

	if (!bFoundProjection)
	{
		return false;
	}

	double BestAnchorDistance = MaxSnapDistanceM;
	for (const TPair<FString, FAeroPedAnchorRuntime>& Pair : AnchorsById)
	{
		const double AnchorDistance = FVector::Distance(OutProjectedEnuM, Pair.Value.PositionEnuM);
		if (AnchorDistance <= BestAnchorDistance)
		{
			BestAnchorDistance = AnchorDistance;
			OutAnchorId = Pair.Key;
		}
	}

	return true;
}

bool UAeroPedNavSemanticSubsystem::ProjectWorldPointToGroundDetailed(const FVector& InputWorldCm, FVector& OutProjectedWorldCm, FVector& OutSurfaceNormalWorld, FString& OutAnchorId) const
{
	const FVector InputEnuM = (InputWorldCm - CurrentWorldOriginCm) / 100.0;
	FVector ProjectedEnuM = InputEnuM;
	FVector SurfaceNormalEnu = FVector::UpVector;
	if (!ProjectPointToGroundDetailed(InputEnuM, ProjectedEnuM, SurfaceNormalEnu, OutAnchorId))
	{
		return false;
	}

	OutProjectedWorldCm = CurrentWorldOriginCm + ProjectedEnuM * 100.0;
	OutSurfaceNormalWorld = SurfaceNormalEnu.GetSafeNormal();
	if (OutSurfaceNormalWorld.IsNearlyZero())
	{
		OutSurfaceNormalWorld = FVector::UpVector;
	}
	return true;
}

bool UAeroPedNavSemanticSubsystem::ProjectWorldPointToGround(const FVector& InputWorldCm, FVector& OutProjectedWorldCm, FString& OutAnchorId) const
{
	FVector SurfaceNormalWorld = FVector::UpVector;
	return ProjectWorldPointToGroundDetailed(InputWorldCm, OutProjectedWorldCm, SurfaceNormalWorld, OutAnchorId);
}

bool UAeroPedNavSemanticSubsystem::ProjectPointToGroundDetailed(const FVector& InputEnuM, FVector& OutProjectedEnuM, FVector& OutSurfaceNormalEnu, FString& OutAnchorId) const
{
	OutProjectedEnuM = InputEnuM;
	OutSurfaceNormalEnu = FVector::UpVector;
	OutAnchorId.Reset();

	const UWorld* World = GetWorld();
	bool bHasTraceProjection = false;
	FVector TraceProjectedEnuM = InputEnuM;
	FVector TraceSurfaceNormalEnu = FVector::UpVector;
	if (SampleGroundPointFromWorld(World, CurrentWorldOriginCm, InputEnuM, GroundTraceHalfHeightM, TraceProjectedEnuM, TraceSurfaceNormalEnu))
	{
		bHasTraceProjection = true;
		OutProjectedEnuM = TraceProjectedEnuM;
		OutSurfaceNormalEnu = TraceSurfaceNormalEnu;
	}

	FString SemanticAnchorId;
	FVector SemanticProjectedEnuM = bHasTraceProjection ? TraceProjectedEnuM : InputEnuM;
	FVector SemanticSurfaceNormalEnu = bHasTraceProjection ? TraceSurfaceNormalEnu : FVector::UpVector;
	if (TryProjectPointOntoSemanticEdges(
			OutgoingEdgesByAnchorId,
			AnchorsById,
			InputEnuM,
			bHasTraceProjection ? TraceProjectedEnuM : InputEnuM,
			MaxSnapDistanceM,
			SemanticProjectedEnuM,
			SemanticSurfaceNormalEnu,
			SemanticAnchorId))
	{
		FVector SemanticTraceProjectedEnuM = SemanticProjectedEnuM;
		FVector SemanticTraceNormalEnu = SemanticSurfaceNormalEnu;
		const FVector SemanticTraceProbeEnuM(SemanticProjectedEnuM.X, SemanticProjectedEnuM.Y, InputEnuM.Z);
		if (SampleGroundPointFromWorld(World, CurrentWorldOriginCm, SemanticTraceProbeEnuM, GroundTraceHalfHeightM, SemanticTraceProjectedEnuM, SemanticTraceNormalEnu))
		{
			OutProjectedEnuM = SemanticTraceProjectedEnuM;
			OutSurfaceNormalEnu = SemanticTraceNormalEnu;
			OutAnchorId = SemanticAnchorId;
			return true;
		}

		if (!bHasTraceProjection || FMath::Abs(SemanticProjectedEnuM.Z - TraceProjectedEnuM.Z) <= MaxSemanticGroundZMismatchM)
		{
			OutProjectedEnuM = SemanticProjectedEnuM;
			OutSurfaceNormalEnu = SemanticSurfaceNormalEnu;
			OutAnchorId = SemanticAnchorId;
			return true;
		}
	}

	double BestDistance = MaxSnapDistanceM;
	for (const TPair<FString, FAeroPedAnchorRuntime>& Pair : AnchorsById)
	{
		const double Distance = FVector2D::Distance(
			FVector2D(InputEnuM.X, InputEnuM.Y),
			FVector2D(Pair.Value.PositionEnuM.X, Pair.Value.PositionEnuM.Y));
		if (Distance <= BestDistance)
		{
			BestDistance = Distance;
			OutAnchorId = Pair.Key;
			FVector AnchorProjectedEnuM = bHasTraceProjection
				? FVector(Pair.Value.PositionEnuM.X, Pair.Value.PositionEnuM.Y, TraceProjectedEnuM.Z)
				: Pair.Value.PositionEnuM;
			FVector AnchorSurfaceNormalEnu = Pair.Value.SurfaceNormalEnu.IsNearlyZero() ? OutSurfaceNormalEnu : Pair.Value.SurfaceNormalEnu;
			const FVector AnchorTraceProbeEnuM(Pair.Value.PositionEnuM.X, Pair.Value.PositionEnuM.Y, InputEnuM.Z);
			if (SampleGroundPointFromWorld(World, CurrentWorldOriginCm, AnchorTraceProbeEnuM, GroundTraceHalfHeightM, AnchorProjectedEnuM, AnchorSurfaceNormalEnu))
			{
				OutProjectedEnuM = AnchorProjectedEnuM;
				OutSurfaceNormalEnu = AnchorSurfaceNormalEnu.IsNearlyZero() ? FVector::UpVector : AnchorSurfaceNormalEnu;
				return true;
			}

			if (bHasTraceProjection)
			{
				OutProjectedEnuM = TraceProjectedEnuM;
				OutSurfaceNormalEnu = TraceSurfaceNormalEnu;
				return true;
			}

			OutProjectedEnuM = AnchorProjectedEnuM;
			OutSurfaceNormalEnu = AnchorSurfaceNormalEnu.IsNearlyZero() ? FVector::UpVector : AnchorSurfaceNormalEnu;
			return true;
		}
	}

	if (bHasTraceProjection)
	{
		OutProjectedEnuM = TraceProjectedEnuM;
		OutSurfaceNormalEnu = TraceSurfaceNormalEnu;
		return true;
	}

	return false;
}

bool UAeroPedNavSemanticSubsystem::ProjectPointToGround(const FVector& InputEnuM, FVector& OutProjectedEnuM, FString& OutAnchorId) const
{
	FVector SurfaceNormalEnu = FVector::UpVector;
	return ProjectPointToGroundDetailed(InputEnuM, OutProjectedEnuM, SurfaceNormalEnu, OutAnchorId);
}
