#include "AeroBridgeWorldSubsystem.h"

#include "AeroAssetPlacementSubsystem.h"
#include "AeroFixedWorldCaptureCamera.h"
#include "AeroRuntimeOrchestrationSubsystem.h"
#include "AeroFeedbackSubsystem.h"
#include "AeroBridgeRuntimeSettings.h"
#include "AeroSemanticStencil.h"
#include "AeroPedNavSemanticSubsystem.h"
#include "AeroSceneSyncSubsystem.h"
#include "AeroWeatherRenderSubsystem.h"
#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "GameFramework/Actor.h"
#include "GroundPlacementUtils.h"
#include "PedestrianRuntimeSettings.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "SumoRoadTopologyQuery.h"

DEFINE_LOG_CATEGORY_STATIC(LogAeroBridgeWorld, Log, All);

namespace
{
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

FString SerializeJsonObject(const TSharedPtr<FJsonObject>& Object)
{
	if (!Object.IsValid())
	{
		return TEXT("{}");
	}

	FString Output;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	FJsonSerializer::Serialize(Object.ToSharedRef(), Writer);
	return Output;
}

void SetStringArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const TArray<FString>& Values)
{
	TArray<TSharedPtr<FJsonValue>> JsonValues;
	for (const FString& Value : Values)
	{
		JsonValues.Add(MakeShared<FJsonValueString>(Value));
	}
	Object->SetArrayField(FieldName, JsonValues);
}

void SetVectorArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& Value)
{
	TArray<TSharedPtr<FJsonValue>> JsonValues;
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.X));
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.Y));
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.Z));
	Object->SetArrayField(FieldName, JsonValues);
}

void SetRotatorObjectField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FRotator& Value)
{
	TSharedPtr<FJsonObject> RotatorObject = MakeShared<FJsonObject>();
	RotatorObject->SetNumberField(TEXT("pitch_deg"), Value.Pitch);
	RotatorObject->SetNumberField(TEXT("yaw_deg"), Value.Yaw);
	RotatorObject->SetNumberField(TEXT("roll_deg"), Value.Roll);
	Object->SetObjectField(FieldName, RotatorObject);
}

TSharedPtr<FJsonObject> MakeClassIdToNamePayload(const TMap<uint8, FString>& Values)
{
	TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
	TArray<uint8> Keys;
	Values.GetKeys(Keys);
	Keys.Sort();
	for (const uint8 Key : Keys)
	{
		Object->SetStringField(FString::FromInt(Key), Values[Key]);
	}
	return Object;
}

TSharedPtr<FJsonObject> MakeClassIdHistogramPayload(const TMap<uint8, int32>& Values)
{
	TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
	TArray<uint8> Keys;
	Values.GetKeys(Keys);
	Keys.Sort();
	for (const uint8 Key : Keys)
	{
		Object->SetNumberField(FString::FromInt(Key), Values[Key]);
	}
	return Object;
}

bool TryReadVectorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector, double DefaultZ = 0.0)
{
	if (!Object.IsValid())
	{
		return false;
	}

	if (Object->HasTypedField<EJson::Array>(FieldName))
	{
		const TArray<TSharedPtr<FJsonValue>>& Values = Object->GetArrayField(FieldName);
		if (Values.Num() >= 2)
		{
			OutVector.X = Values[0]->AsNumber();
			OutVector.Y = Values[1]->AsNumber();
			OutVector.Z = Values.Num() > 2 ? Values[2]->AsNumber() : DefaultZ;
			return true;
		}
	}

	if (Object->HasTypedField<EJson::Object>(FieldName))
	{
		const TSharedPtr<FJsonObject> VectorObject = Object->GetObjectField(FieldName);
		if (VectorObject.IsValid())
		{
			const bool bHasX = VectorObject->HasField(TEXT("x")) || VectorObject->HasField(TEXT("east_m"));
			const bool bHasY = VectorObject->HasField(TEXT("y")) || VectorObject->HasField(TEXT("north_m"));
			if (!bHasX || !bHasY)
			{
				return false;
			}

			const double X = VectorObject->HasField(TEXT("x")) ? VectorObject->GetNumberField(TEXT("x")) : VectorObject->GetNumberField(TEXT("east_m"));
			const double Y = VectorObject->HasField(TEXT("y")) ? VectorObject->GetNumberField(TEXT("y")) : VectorObject->GetNumberField(TEXT("north_m"));
			const double Z = VectorObject->HasField(TEXT("z"))
				? VectorObject->GetNumberField(TEXT("z"))
				: (VectorObject->HasField(TEXT("up_m")) ? VectorObject->GetNumberField(TEXT("up_m")) : DefaultZ);
			OutVector = FVector(X, Y, Z);
			return true;
		}
	}

	return false;
}

bool TryReadBoolField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, bool& OutValue)
{
	if (!Object.IsValid() || !Object->HasField(FieldName))
	{
		return false;
	}

	OutValue = Object->GetBoolField(FieldName);
	return true;
}

bool TryReadRotatorFieldFromObject(const TSharedPtr<FJsonObject>& Object, FRotator& OutRotator)
{
	if (!Object.IsValid())
	{
		return false;
	}

	double Pitch = 0.0;
	double Yaw = 0.0;
	double Roll = 0.0;
	bool bHasAny = false;
	if (Object->TryGetNumberField(TEXT("pitch_deg"), Pitch) || Object->TryGetNumberField(TEXT("pitch"), Pitch))
	{
		bHasAny = true;
	}
	if (Object->TryGetNumberField(TEXT("yaw_deg"), Yaw) || Object->TryGetNumberField(TEXT("yaw"), Yaw))
	{
		bHasAny = true;
	}
	if (Object->TryGetNumberField(TEXT("roll_deg"), Roll) || Object->TryGetNumberField(TEXT("roll"), Roll))
	{
		bHasAny = true;
	}
	if (!bHasAny)
	{
		return false;
	}

	OutRotator = FRotator(static_cast<float>(Pitch), static_cast<float>(Yaw), static_cast<float>(Roll));
	return true;
}

bool TryReadRotatorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FRotator& OutRotator)
{
	return Object.IsValid() && Object->HasTypedField<EJson::Object>(FieldName)
		? TryReadRotatorFieldFromObject(Object->GetObjectField(FieldName), OutRotator)
		: false;
}

FString ToRuntimeMoveStateJsonString(const EAeroRuntimeMoveState State)
{
	switch (State)
	{
	case EAeroRuntimeMoveState::Running:
		return TEXT("running");
	case EAeroRuntimeMoveState::Succeeded:
		return TEXT("succeeded");
	case EAeroRuntimeMoveState::Failed:
		return TEXT("failed");
	case EAeroRuntimeMoveState::Cancelled:
		return TEXT("cancelled");
	default:
		return TEXT("idle");
	}
}

bool TryReadPosePositionEnuField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionEnuM)
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> PoseObject = Object->GetObjectField(FieldName);
	if (!PoseObject.IsValid())
	{
		return false;
	}

	return TryReadVectorField(PoseObject, TEXT("position_m"), OutPositionEnuM) ||
		TryReadVectorField(PoseObject, TEXT("position_enu_m"), OutPositionEnuM);
}

UAeroRuntimeOrchestrationSubsystem* ResolveRuntimeOrchestrationSubsystem(UWorld* World)
{
	return World != nullptr ? World->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
}

FVector ReadWorldOriginCm(const TSharedPtr<FJsonObject>& MapContext)
{
	FVector WorldOriginCm = FVector::ZeroVector;
	TryReadVectorField(MapContext, TEXT("world_origin_cm"), WorldOriginCm);
	return WorldOriginCm;
}

bool TryResolveWorldPointFromPayload(
	const TSharedPtr<FJsonObject>& Payload,
	const FString& WorldFieldName,
	const FString& EnuFieldName,
	const FVector& WorldOriginCm,
	FVector& OutWorldCm)
{
	if (TryReadVectorField(Payload, WorldFieldName, OutWorldCm))
	{
		return true;
	}

	FVector EnuM = FVector::ZeroVector;
	if (!TryReadVectorField(Payload, EnuFieldName, EnuM))
	{
		return false;
	}

	OutWorldCm = WorldOriginCm + EnuM * 100.0;
	return true;
}

bool SnapWorldPointToGround(UWorld* World, FVector& InOutWorldCm, bool* bOutUsedRoadTopology = nullptr, FString* OutGroundSource = nullptr)
{
	if (bOutUsedRoadTopology != nullptr)
	{
		*bOutUsedRoadTopology = false;
	}
	if (OutGroundSource != nullptr)
	{
		OutGroundSource->Reset();
	}

	FVector CandidateWorldCm = InOutWorldCm;
	bool bUsedRoadTopology = false;

	if (World != nullptr)
	{
		FSumoNearestLaneSample RoadSample;
		FString RoadError;
		if (FSumoRoadTopologyQuery::FindNearestRoadSample(World, InOutWorldCm, RoadSample, RoadError))
		{
			const FVector RoadWorldCm = RoadSample.WorldTransform.GetLocation();
			CandidateWorldCm.X = RoadWorldCm.X;
			CandidateWorldCm.Y = RoadWorldCm.Y;
			bUsedRoadTopology = true;
			if (bOutUsedRoadTopology != nullptr)
			{
				*bOutUsedRoadTopology = true;
			}
		}
	}

	AeroGroundPlacement::FResolvedGroundPlacement Placement;
	if (!AeroGroundPlacement::ResolveGroundPlacement(World, CandidateWorldCm, Placement))
	{
		return false;
	}

	InOutWorldCm = Placement.GroundWorldCm;
	if (bUsedRoadTopology)
	{
		InOutWorldCm.X = CandidateWorldCm.X;
		InOutWorldCm.Y = CandidateWorldCm.Y;
	}
	if (OutGroundSource != nullptr)
	{
		*OutGroundSource = bUsedRoadTopology
			? FString::Printf(TEXT("road_xy+%s"), Placement.Source.IsEmpty() ? TEXT("trace") : *Placement.Source)
			: Placement.Source;
	}
	return true;
}

bool SnapWorldPointToGroundPreserveXY(UWorld* World, FVector& InOutWorldCm, float VerticalOffsetCm = 0.0f, FString* OutGroundSource = nullptr)
{
	if (OutGroundSource != nullptr)
	{
		OutGroundSource->Reset();
	}

	const FVector RequestedWorldCm = InOutWorldCm;
	FVector ProjectedWorldCm = RequestedWorldCm;
	FVector SurfaceNormalWorld = FVector::UpVector;
	if (AeroGroundPlacement::TryProjectWorldPointToGround(World, RequestedWorldCm, ProjectedWorldCm, &SurfaceNormalWorld))
	{
		InOutWorldCm = RequestedWorldCm;
		InOutWorldCm.Z = ProjectedWorldCm.Z + VerticalOffsetCm;
		if (OutGroundSource != nullptr)
		{
			*OutGroundSource = TEXT("trace_preserve_xy");
		}
		return true;
	}

	AeroGroundPlacement::FResolvedGroundPlacement Placement;
	if (AeroGroundPlacement::ResolveGroundPlacement(World, RequestedWorldCm, Placement))
	{
		InOutWorldCm = RequestedWorldCm;
		InOutWorldCm.Z = Placement.GroundWorldCm.Z + VerticalOffsetCm;
		if (OutGroundSource != nullptr)
		{
			if (Placement.Source.IsEmpty())
			{
				*OutGroundSource = TEXT("ground_z_preserve_xy");
			}
			else
			{
				*OutGroundSource = FString::Printf(TEXT("%s_z_preserve_xy"), *Placement.Source);
			}
		}
		return true;
	}

	return false;
}

float ReadEnuVerticalOffsetCm(const TSharedPtr<FJsonObject>& Payload, const FString& FieldName)
{
	FVector EnuM = FVector::ZeroVector;
	if (TryReadVectorField(Payload, FieldName, EnuM))
	{
		return static_cast<float>(EnuM.Z * 100.0);
	}
	return 0.0f;
}

TSharedPtr<FJsonObject> MakeVectorPayloadField(const FVector& VectorValue)
{
	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetNumberField(TEXT("x"), VectorValue.X);
	Result->SetNumberField(TEXT("y"), VectorValue.Y);
	Result->SetNumberField(TEXT("z"), VectorValue.Z);
	return Result;
}

TSharedPtr<FJsonObject> CrowdSpawnResultToJson(const FCrowdSpawnResult& Result)
{
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("group_id"), Result.GroupId.ToString());
	Payload->SetNumberField(TEXT("skipped_count"), Result.SkippedCount);
	Payload->SetNumberField(TEXT("seed"), Result.Seed);
	SetStringArrayField(Payload, TEXT("spawned_ids"), Result.SpawnedIds);
	return Payload;
}

ECrowdYawPolicy ParseCrowdYawPolicy(const FString& Value)
{
	return Value.Equals(TEXT("fixed"), ESearchCase::IgnoreCase) ? ECrowdYawPolicy::Fixed : ECrowdYawPolicy::Random;
}

}

bool UAeroBridgeWorldSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroBridgeWorldSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
}

TSharedPtr<FJsonObject> UAeroBridgeWorldSubsystem::ParseRequestEnvelope(const FString& RequestJson, FString& OutRequestId, FString& OutMapId, FString& OutError) const
{
	OutRequestId = FGuid::NewGuid().ToString(EGuidFormats::Digits);
	OutMapId = CurrentMapId;

	TSharedPtr<FJsonObject> RootObject = MakeShared<FJsonObject>();
	if (!RequestJson.TrimStartAndEnd().IsEmpty())
	{
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(RequestJson);
		if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
		{
			OutError = TEXT("Request JSON is invalid.");
			return nullptr;
		}
	}

	RootObject->TryGetStringField(TEXT("request_id"), OutRequestId);
	RootObject->TryGetStringField(TEXT("map_id"), OutMapId);

	TSharedPtr<FJsonObject> PayloadObject = RootObject;
	if (RootObject->HasTypedField<EJson::Object>(TEXT("payload")))
	{
		PayloadObject = RootObject->GetObjectField(TEXT("payload"));
	}

	if (PayloadObject.IsValid() && OutMapId.IsEmpty())
	{
		PayloadObject->TryGetStringField(TEXT("map_id"), OutMapId);
	}

	return PayloadObject;
}

FString UAeroBridgeWorldSubsystem::MakeSuccessResponse(const FString& Op, const FString& RequestId, const FString& MapId, const TSharedPtr<FJsonObject>& Payload) const
{
	TSharedPtr<FJsonObject> Response = MakeShared<FJsonObject>();
	Response->SetStringField(TEXT("api_version"), TEXT("1.0"));
	Response->SetStringField(TEXT("map_id"), MapId);
	Response->SetStringField(TEXT("request_id"), RequestId);
	Response->SetStringField(TEXT("op"), Op);
	Response->SetStringField(TEXT("status"), TEXT("ok"));
	Response->SetObjectField(TEXT("payload"), Payload.IsValid() ? Payload : MakeShared<FJsonObject>());
	return SerializeJsonObject(Response);
}

FString UAeroBridgeWorldSubsystem::MakeErrorResponse(const FString& Op, const FString& RequestId, const FString& MapId, const FString& ErrorMessage) const
{
	TSharedPtr<FJsonObject> ErrorObject = MakeShared<FJsonObject>();
	ErrorObject->SetStringField(TEXT("message"), ErrorMessage);

	TSharedPtr<FJsonObject> Response = MakeShared<FJsonObject>();
	Response->SetStringField(TEXT("api_version"), TEXT("1.0"));
	Response->SetStringField(TEXT("map_id"), MapId);
	Response->SetStringField(TEXT("request_id"), RequestId);
	Response->SetStringField(TEXT("op"), Op);
	Response->SetStringField(TEXT("status"), TEXT("error"));
	Response->SetObjectField(TEXT("payload"), MakeShared<FJsonObject>());
	Response->SetObjectField(TEXT("error"), ErrorObject);
	return SerializeJsonObject(Response);
}

FString UAeroBridgeWorldSubsystem::ResolveRelativePath(const FString& MaybeRelativePath) const
{
	if (MaybeRelativePath.IsEmpty())
	{
		return FString();
	}

	if (FPaths::IsRelative(MaybeRelativePath))
	{
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir(), MaybeRelativePath);
	}

	return MaybeRelativePath;
}

FString UAeroBridgeWorldSubsystem::ResolveMapPath(const FString& MapId, const FString& FileName) const
{
	const UAeroBridgeRuntimeSettings* Settings = GetDefault<UAeroBridgeRuntimeSettings>();
	const FString MapsRoot = ResolveRelativePath(Settings->MapsRelativeRoot);
	return FPaths::Combine(MapsRoot, MapId, FileName);
}

bool UAeroBridgeWorldSubsystem::LoadContextByMapId(const FString& MapId, FString& OutError)
{
	if (MapId.IsEmpty())
	{
		OutError = TEXT("LoadContext requires a non-empty map_id.");
		return false;
	}

	TSharedPtr<FJsonObject> MapContext;
	if (!LoadJsonObjectFromFile(ResolveMapPath(MapId, TEXT("map_context.json")), MapContext, OutError))
	{
		return false;
	}

	return ApplyLoadedMapContext(MapId, MapContext, OutError);
}

bool UAeroBridgeWorldSubsystem::ApplyLoadedMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext, FString& OutError)
{
	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>();
	UAeroPedNavSemanticSubsystem* PedSubsystem = GetWorld()->GetSubsystem<UAeroPedNavSemanticSubsystem>();
	UAeroSceneSyncSubsystem* SceneSyncSubsystem = GetWorld()->GetSubsystem<UAeroSceneSyncSubsystem>();
	UAeroWeatherRenderSubsystem* WeatherSubsystem = GetWorld()->GetSubsystem<UAeroWeatherRenderSubsystem>();
	if (AssetSubsystem == nullptr || FeedbackSubsystem == nullptr || PedSubsystem == nullptr || SceneSyncSubsystem == nullptr || WeatherSubsystem == nullptr)
	{
		OutError = TEXT("One or more Aero subsystems are unavailable.");
		return false;
	}

	const UAeroBridgeRuntimeSettings* Settings = GetDefault<UAeroBridgeRuntimeSettings>();
	AssetSubsystem->SetMapContext(MapId, MapContext);
	PedSubsystem->SetMapContext(MapId, MapContext);
	FeedbackSubsystem->ResetEpisode(TEXT(""));
	if (MapContext.IsValid())
	{
		FVector WorldOriginCm = FVector::ZeroVector;
		if (MapContext->HasTypedField<EJson::Array>(TEXT("world_origin_cm")))
		{
			const TArray<TSharedPtr<FJsonValue>>& Values = MapContext->GetArrayField(TEXT("world_origin_cm"));
			if (Values.Num() >= 3)
			{
				WorldOriginCm.X = Values[0]->AsNumber();
				WorldOriginCm.Y = Values[1]->AsNumber();
				WorldOriginCm.Z = Values[2]->AsNumber();
			}
		}
		FeedbackSubsystem->SetWorldOriginCm(WorldOriginCm);
	}

	if (!AssetSubsystem->LoadAssetCatalog(ResolveRelativePath(Settings->AssetCatalogRelativePath), OutError))
	{
		return false;
	}
	if (!WeatherSubsystem->LoadProfiles(ResolveRelativePath(Settings->WeatherProfilesRelativePath), OutError))
	{
		return false;
	}
	if (!AssetSubsystem->LoadScenarioObjects(ResolveMapPath(MapId, TEXT("scenario_objects.json")), OutError))
	{
		return false;
	}

	const FString SemanticSourcePath = ResolveMapPath(MapId, TEXT("ped_nav_semantic.source.json"));
	const FString SemanticBundlePath = ResolveMapPath(MapId, TEXT("ped_nav_semantic.bundle.json"));
	if (FPaths::FileExists(SemanticSourcePath))
	{
		PedSubsystem->LoadSemanticSource(SemanticSourcePath, OutError);
		if (!PedSubsystem->CompileSemanticBundle(SemanticSourcePath, SemanticBundlePath, OutError))
		{
			return false;
		}
	}
	if (FPaths::FileExists(SemanticBundlePath))
	{
		if (!PedSubsystem->LoadSemanticBundle(SemanticBundlePath, OutError))
		{
			return false;
		}
	}

	SceneSyncSubsystem->ResetSyncState();
	CurrentMapId = MapId;
	CurrentMapContext = MapContext;
	return true;
}

FString UAeroBridgeWorldSubsystem::HandleDescribeCapabilities(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroDescribeCapabilities"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	SetStringArrayField(
		ResponsePayload,
		TEXT("operations"),
		{
			TEXT("simAeroDescribeCapabilities"),
			TEXT("simAeroLoadContext"),
			TEXT("simAeroReloadConfig"),
			TEXT("simAeroApplyFrame"),
			TEXT("simAeroPollFeedback"),
			TEXT("simAeroPedSpawn"),
			TEXT("simAeroPedReset"),
			TEXT("simAeroPedSetTarget"),
			TEXT("simAeroPedObserve"),
			TEXT("simAeroPedPlayAnimation"),
			TEXT("simAeroPedCommitCross"),
			TEXT("simAeroPedStop"),
			TEXT("simAeroPedSetVariant"),
			TEXT("simAeroPedRelease"),
			TEXT("simAeroPedSpawnCrowd"),
			TEXT("simAeroPedClearCrowd"),
			TEXT("simAeroPedRespawnCrowd"),
			TEXT("simAeroSpawnAsset"),
			TEXT("simAeroMoveAsset"),
			TEXT("simAeroRemoveAsset"),
			TEXT("simAeroCaptureWorldCamera"),
			TEXT("aero.semantic_stencil_audit_json"),
			TEXT("simAeroReserveOccupancy"),
			TEXT("simAeroReleaseOccupancy"),
			TEXT("simAeroQueryNearest"),
			TEXT("simAeroQueryPedPath"),
			TEXT("simAeroProjectGround"),
			TEXT("simAeroQueryPedAnchor"),
			TEXT("simAeroApplyWeather"),
			TEXT("simAeroCreateRuntimeMultirotor"),
			TEXT("simAeroMoveRuntimeMultirotor"),
			TEXT("simAeroGetRuntimeMultirotorStatus"),
			TEXT("simAeroRemoveRuntimeVehicle"),
			TEXT("simAeroGetRuntimeVehiclePose")
		});
	SetStringArrayField(
		ResponsePayload,
		TEXT("config_kinds"),
		{
			TEXT("asset_catalog"),
			TEXT("scenario_objects"),
			TEXT("ped_nav_semantic"),
			TEXT("weather_profiles"),
			TEXT("map_context")
		});
	SetStringArrayField(
		ResponsePayload,
		TEXT("ped_variants"),
		{
			TEXT("adult_male_commuter"),
			TEXT("adult_female_commuter"),
			TEXT("child_crossing"),
			TEXT("elder_observer")
		});
	SetStringArrayField(
		ResponsePayload,
		TEXT("ped_modes"),
		{
			TEXT("idle"),
			TEXT("stop"),
			TEXT("observe"),
			TEXT("start_cross"),
			TEXT("cross")
		});
	SetStringArrayField(
		ResponsePayload,
		TEXT("ped_montage_tags"),
		{
			TEXT("observe"),
			TEXT("start_cross"),
			TEXT("stop")
		});
	ResponsePayload->SetStringField(TEXT("local_frame"), TEXT("ENU"));
	ResponsePayload->SetStringField(TEXT("current_map_id"), CurrentMapId);
	return MakeSuccessResponse(TEXT("simAeroDescribeCapabilities"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleLoadContext(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroLoadContext"), RequestId, MapId, Error);
	}

	if (MapId.IsEmpty())
	{
		Payload->TryGetStringField(TEXT("map_id"), MapId);
	}
	if (!LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroLoadContext"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("map_id"), MapId);
	ResponsePayload->SetStringField(TEXT("map_context_path"), ResolveMapPath(MapId, TEXT("map_context.json")));
	ResponsePayload->SetStringField(TEXT("scenario_objects_path"), ResolveMapPath(MapId, TEXT("scenario_objects.json")));
	ResponsePayload->SetStringField(TEXT("ped_nav_bundle_path"), ResolveMapPath(MapId, TEXT("ped_nav_semantic.bundle.json")));
	if (CurrentMapContext.IsValid())
	{
		FString LocalFrame;
		if (CurrentMapContext->TryGetStringField(TEXT("local_frame"), LocalFrame))
		{
			ResponsePayload->SetStringField(TEXT("local_frame"), LocalFrame);
		}

		FString WorldOriginPolicy;
		if (CurrentMapContext->TryGetStringField(TEXT("world_origin_policy"), WorldOriginPolicy))
		{
			ResponsePayload->SetStringField(TEXT("world_origin_policy"), WorldOriginPolicy);
		}

		FString UELevelName;
		if (CurrentMapContext->TryGetStringField(TEXT("ue_level_name"), UELevelName))
		{
			ResponsePayload->SetStringField(TEXT("ue_level_name"), UELevelName);
		}

		SetVectorArrayField(ResponsePayload, TEXT("world_origin_cm"), ReadWorldOriginCm(CurrentMapContext));
		if (CurrentMapContext->HasTypedField<EJson::Object>(TEXT("geo_reference")))
		{
			ResponsePayload->SetObjectField(TEXT("geo_reference"), CurrentMapContext->GetObjectField(TEXT("geo_reference")));
		}
	}
	return MakeSuccessResponse(TEXT("simAeroLoadContext"), RequestId, MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleReloadConfig(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
	}

	FString Kind;
	FString Path;
	Payload->TryGetStringField(TEXT("kind"), Kind);
	Payload->TryGetStringField(TEXT("path"), Path);
	if (Kind.IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, TEXT("ReloadConfig requires kind."));
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	UAeroPedNavSemanticSubsystem* PedSubsystem = GetWorld()->GetSubsystem<UAeroPedNavSemanticSubsystem>();
	UAeroWeatherRenderSubsystem* WeatherSubsystem = GetWorld()->GetSubsystem<UAeroWeatherRenderSubsystem>();
	if (AssetSubsystem == nullptr || PedSubsystem == nullptr || WeatherSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, TEXT("One or more Aero subsystems are unavailable."));
	}

	const UAeroBridgeRuntimeSettings* Settings = GetDefault<UAeroBridgeRuntimeSettings>();
	if (Kind.Equals(TEXT("asset_catalog"), ESearchCase::IgnoreCase))
	{
		if (Path.IsEmpty())
		{
			Path = ResolveRelativePath(Settings->AssetCatalogRelativePath);
		}
		if (!AssetSubsystem->LoadAssetCatalog(ResolveRelativePath(Path), Error))
		{
			return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
		}
	}
	else if (Kind.Equals(TEXT("weather_profiles"), ESearchCase::IgnoreCase))
	{
		if (Path.IsEmpty())
		{
			Path = ResolveRelativePath(Settings->WeatherProfilesRelativePath);
		}
		if (!WeatherSubsystem->LoadProfiles(ResolveRelativePath(Path), Error))
		{
			return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
		}
	}
	else if (Kind.Equals(TEXT("scenario_objects"), ESearchCase::IgnoreCase))
	{
		if (MapId.IsEmpty())
		{
			MapId = CurrentMapId;
		}
		if (Path.IsEmpty())
		{
			Path = ResolveMapPath(MapId, TEXT("scenario_objects.json"));
		}
		if (!AssetSubsystem->LoadScenarioObjects(ResolveRelativePath(Path), Error))
		{
			return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
		}
	}
	else if (Kind.Equals(TEXT("ped_nav_semantic"), ESearchCase::IgnoreCase))
	{
		if (MapId.IsEmpty())
		{
			MapId = CurrentMapId;
		}
		if (Path.IsEmpty())
		{
			Path = ResolveMapPath(MapId, TEXT("ped_nav_semantic.bundle.json"));
		}
		const FString ResolvedPath = ResolveRelativePath(Path);
		if (ResolvedPath.EndsWith(TEXT(".source.json")))
		{
			const FString BundlePath = ResolvedPath.Replace(TEXT(".source.json"), TEXT(".bundle.json"));
			if (!PedSubsystem->CompileSemanticBundle(ResolvedPath, BundlePath, Error) || !PedSubsystem->LoadSemanticBundle(BundlePath, Error))
			{
				return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
			}
		}
		else if (!PedSubsystem->LoadSemanticBundle(ResolvedPath, Error))
		{
			return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
		}
	}
	else if (Kind.Equals(TEXT("map_context"), ESearchCase::IgnoreCase))
	{
		if (MapId.IsEmpty())
		{
			MapId = CurrentMapId;
		}
		if (Path.IsEmpty())
		{
			Path = ResolveMapPath(MapId, TEXT("map_context.json"));
		}

		TSharedPtr<FJsonObject> MapContext;
		if (!LoadJsonObjectFromFile(ResolveRelativePath(Path), MapContext, Error) || !ApplyLoadedMapContext(MapId, MapContext, Error))
		{
			return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, Error);
		}
	}
	else
	{
		return MakeErrorResponse(TEXT("simAeroReloadConfig"), RequestId, MapId, FString::Printf(TEXT("Unsupported config kind: %s"), *Kind));
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("kind"), Kind);
	ResponsePayload->SetStringField(TEXT("path"), ResolveRelativePath(Path));
	return MakeSuccessResponse(TEXT("simAeroReloadConfig"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleApplyFrame(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroApplyFrame"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroApplyFrame"), RequestId, MapId, Error);
	}

	UAeroSceneSyncSubsystem* SceneSyncSubsystem = GetWorld()->GetSubsystem<UAeroSceneSyncSubsystem>();
	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>();
	if (SceneSyncSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroApplyFrame"), RequestId, MapId, TEXT("AeroSceneSync subsystem unavailable."));
	}
	if (FeedbackSubsystem != nullptr)
	{
		FAeroFrameContext FrameContext;
		double NumberValue = 0.0;
		if (Payload->TryGetNumberField(TEXT("tick"), NumberValue))
		{
			FrameContext.Tick = static_cast<int64>(NumberValue);
		}
		if (Payload->TryGetNumberField(TEXT("frame_id"), NumberValue))
		{
			FrameContext.FrameId = static_cast<int64>(NumberValue);
		}
		if (Payload->TryGetNumberField(TEXT("sample_seq"), NumberValue))
		{
			FrameContext.SampleSeq = static_cast<int64>(NumberValue);
		}
		Payload->TryGetNumberField(TEXT("sim_time_s"), FrameContext.SimTimeS);
		Payload->TryGetStringField(TEXT("episode_id"), FrameContext.EpisodeId);
		FeedbackSubsystem->SetFrameContext(FrameContext);
	}

	TSharedPtr<FJsonObject> Result = SceneSyncSubsystem->ApplyFrame(Payload, Error);
	if (!Result.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroApplyFrame"), RequestId, MapId, Error);
	}
	return MakeSuccessResponse(TEXT("simAeroApplyFrame"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result);
}

FString UAeroBridgeWorldSubsystem::HandlePollFeedback(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPollFeedback"), RequestId, MapId, Error);
	}

	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>();
	if (FeedbackSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPollFeedback"), RequestId, MapId, TEXT("AeroFeedback subsystem unavailable."));
	}

	int64 SinceTick = INDEX_NONE;
	int64 SinceFrameId = INDEX_NONE;
	double SinceValue = 0.0;
	if (Payload->TryGetNumberField(TEXT("since_tick"), SinceValue))
	{
		SinceTick = static_cast<int64>(SinceValue);
	}
	if (Payload->TryGetNumberField(TEXT("since_frame_id"), SinceValue))
	{
		SinceFrameId = static_cast<int64>(SinceValue);
	}

	TArray<FAeroFeedbackEvent> Events;
	int64 UptoTick = 0;
	int64 UptoFrameId = 0;
	FString EpisodeId;
	if (SinceTick != INDEX_NONE)
	{
		FeedbackSubsystem->PollFeedbackSinceTick(SinceTick, Events, UptoTick, UptoFrameId, EpisodeId);
	}
	else if (SinceFrameId != INDEX_NONE)
	{
		FeedbackSubsystem->PollFeedbackSinceFrame(SinceFrameId, Events, UptoTick, UptoFrameId, EpisodeId);
	}
	else
	{
		FeedbackSubsystem->PollAllFeedback(Events, UptoTick, UptoFrameId, EpisodeId);
	}

	TArray<TSharedPtr<FJsonValue>> JsonEvents;
	for (const FAeroFeedbackEvent& Event : Events)
	{
		JsonEvents.Add(MakeShared<FJsonValueObject>(AeroFeedbackEventToJson(Event)));
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetArrayField(TEXT("events"), JsonEvents);
	ResponsePayload->SetNumberField(TEXT("upto_tick"), UptoTick);
	ResponsePayload->SetNumberField(TEXT("upto_frame_id"), UptoFrameId);
	ResponsePayload->SetStringField(TEXT("episode_id"), EpisodeId);
	return MakeSuccessResponse(TEXT("simAeroPollFeedback"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedSpawn(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, TEXT("ped_id is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector SpawnWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, SpawnWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, TEXT("position_world_cm or position_enu_m is required."));
	}

	bool bSnapToGround = true;
	TryReadBoolField(Payload, TEXT("snap_to_ground"), bSnapToGround);
	bool bPreserveXY = true;
	TryReadBoolField(Payload, TEXT("preserve_xy"), bPreserveXY);
	bool bUseProvidedGroundPoint = !bSnapToGround;
	FString GroundSource;
	if (bSnapToGround)
	{
		const float VerticalOffsetCm = ReadEnuVerticalOffsetCm(Payload, TEXT("position_enu_m"));
		bUseProvidedGroundPoint = bPreserveXY
			? SnapWorldPointToGroundPreserveXY(GetWorld(), SpawnWorldCm, VerticalOffsetCm, &GroundSource)
			: SnapWorldPointToGround(GetWorld(), SpawnWorldCm, nullptr, &GroundSource);
		if (!GroundSource.IsEmpty())
		{
			UE_LOG(
				LogAeroBridgeWorld,
				Log,
				TEXT("simAeroPedSpawn grounded: map_id='%s' ped_id='%s' world='%s' source='%s' preserve_xy=%s."),
				*(MapId.IsEmpty() ? CurrentMapId : MapId),
				*PedId,
				*SpawnWorldCm.ToString(),
				*GroundSource,
				bPreserveXY ? TEXT("true") : TEXT("false"));
		}
	}

	double YawDeg = 0.0;
	Payload->TryGetNumberField(TEXT("yaw_deg"), YawDeg);
	FString VariantIdString;
	Payload->TryGetStringField(TEXT("variant_id"), VariantIdString);
	const FName VariantId = VariantIdString.TrimStartAndEnd().IsEmpty() ? NAME_None : FName(*VariantIdString.TrimStartAndEnd());
	if (!RuntimeSubsystem->SpawnPedestrian(PedId, SpawnWorldCm, static_cast<float>(YawDeg), VariantId, Error, bUseProvidedGroundPoint))
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawn"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetNumberField(TEXT("yaw_deg"), YawDeg);
	ResponsePayload->SetObjectField(TEXT("position_world_cm"), MakeVectorPayloadField(SpawnWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("position_enu_m"), (SpawnWorldCm - WorldOriginCm) / 100.0);
	ResponsePayload->SetBoolField(TEXT("used_provided_ground_point"), bUseProvidedGroundPoint);
	ResponsePayload->SetBoolField(TEXT("preserve_xy"), bPreserveXY);
	if (!GroundSource.IsEmpty())
	{
		ResponsePayload->SetStringField(TEXT("ground_source"), GroundSource);
	}
	if (!VariantId.IsNone())
	{
		ResponsePayload->SetStringField(TEXT("variant_id"), VariantId.ToString());
	}
	return MakeSuccessResponse(TEXT("simAeroPedSpawn"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedReset(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, TEXT("ped_id is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector ResetWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, ResetWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, TEXT("position_world_cm or position_enu_m is required."));
	}

	bool bSnapToGround = true;
	TryReadBoolField(Payload, TEXT("snap_to_ground"), bSnapToGround);
	bool bPreserveXY = true;
	TryReadBoolField(Payload, TEXT("preserve_xy"), bPreserveXY);
	bool bUseProvidedGroundPoint = !bSnapToGround;
	FString GroundSource;
	if (bSnapToGround)
	{
		const float VerticalOffsetCm = ReadEnuVerticalOffsetCm(Payload, TEXT("position_enu_m"));
		bUseProvidedGroundPoint = bPreserveXY
			? SnapWorldPointToGroundPreserveXY(GetWorld(), ResetWorldCm, VerticalOffsetCm, &GroundSource)
			: SnapWorldPointToGround(GetWorld(), ResetWorldCm, nullptr, &GroundSource);
		if (!GroundSource.IsEmpty())
		{
			UE_LOG(
				LogAeroBridgeWorld,
				Log,
				TEXT("simAeroPedReset grounded: map_id='%s' ped_id='%s' world='%s' source='%s' preserve_xy=%s."),
				*(MapId.IsEmpty() ? CurrentMapId : MapId),
				*PedId,
				*ResetWorldCm.ToString(),
				*GroundSource,
				bPreserveXY ? TEXT("true") : TEXT("false"));
		}
	}

	double YawDeg = 0.0;
	Payload->TryGetNumberField(TEXT("yaw_deg"), YawDeg);
	bool bFramePose = false;
	TryReadBoolField(Payload, TEXT("frame_pose"), bFramePose);
	bool bWalking = false;
	TryReadBoolField(Payload, TEXT("walking"), bWalking);
	double SpeedCmPerSec = 0.0;
	Payload->TryGetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);

	const bool bSuccess = bFramePose
		? RuntimeSubsystem->SetPedestrianFramePose(
			PedId,
			ResetWorldCm,
			static_cast<float>(YawDeg),
			bWalking,
			static_cast<float>(SpeedCmPerSec),
			Error,
			bUseProvidedGroundPoint)
		: RuntimeSubsystem->ResetPedestrian(PedId, ResetWorldCm, static_cast<float>(YawDeg), Error, bUseProvidedGroundPoint);
	if (!bSuccess)
	{
		return MakeErrorResponse(TEXT("simAeroPedReset"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetNumberField(TEXT("yaw_deg"), YawDeg);
	ResponsePayload->SetBoolField(TEXT("frame_pose"), bFramePose);
	ResponsePayload->SetBoolField(TEXT("walking"), bWalking);
	ResponsePayload->SetBoolField(TEXT("preserve_xy"), bPreserveXY);
	ResponsePayload->SetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);
	ResponsePayload->SetObjectField(TEXT("position_world_cm"), MakeVectorPayloadField(ResetWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("position_enu_m"), (ResetWorldCm - WorldOriginCm) / 100.0);
	ResponsePayload->SetBoolField(TEXT("used_provided_ground_point"), bUseProvidedGroundPoint);
	if (!GroundSource.IsEmpty())
	{
		ResponsePayload->SetStringField(TEXT("ground_source"), GroundSource);
	}
	return MakeSuccessResponse(TEXT("simAeroPedReset"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedSetTarget(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, TEXT("ped_id is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector TargetWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("target_world_cm"), TEXT("target_enu_m"), WorldOriginCm, TargetWorldCm) &&
		!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, TargetWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, TEXT("target_world_cm or target_enu_m is required."));
	}

	bool bSnapToGround = true;
	TryReadBoolField(Payload, TEXT("snap_to_ground"), bSnapToGround);
	FString GroundSource;
	if (bSnapToGround)
	{
		SnapWorldPointToGround(GetWorld(), TargetWorldCm, nullptr, &GroundSource);
	}

	double SpeedCmPerSec = 0.0;
	Payload->TryGetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);
	if (!RuntimeSubsystem->SetPedestrianTarget(PedId, TargetWorldCm, static_cast<float>(SpeedCmPerSec), Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);
	ResponsePayload->SetObjectField(TEXT("target_world_cm"), MakeVectorPayloadField(TargetWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("target_enu_m"), (TargetWorldCm - WorldOriginCm) / 100.0);
	if (!GroundSource.IsEmpty())
	{
		ResponsePayload->SetStringField(TEXT("ground_source"), GroundSource);
	}
	return MakeSuccessResponse(TEXT("simAeroPedSetTarget"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedObserve(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedObserve"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedObserve"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedObserve"), RequestId, MapId, TEXT("ped_id is required."));
	}

	FString StartSection;
	Payload->TryGetStringField(TEXT("start_section"), StartSection);
	if (!RuntimeSubsystem->ObservePedestrian(PedId, FName(*StartSection.TrimStartAndEnd()), Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedObserve"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	if (!StartSection.TrimStartAndEnd().IsEmpty())
	{
		ResponsePayload->SetStringField(TEXT("start_section"), StartSection.TrimStartAndEnd());
	}
	return MakeSuccessResponse(TEXT("simAeroPedObserve"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedPlayAnimation(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId, TEXT("ped_id is required."));
	}

	FString AnimationAssetPath;
	Payload->TryGetStringField(TEXT("animation_asset_path"), AnimationAssetPath);
	if (AnimationAssetPath.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId, TEXT("animation_asset_path is required."));
	}

	FString StartSection;
	Payload->TryGetStringField(TEXT("start_section"), StartSection);

	double PlayRate = 1.0;
	Payload->TryGetNumberField(TEXT("play_rate"), PlayRate);

	double LoopCountDouble = 1.0;
	Payload->TryGetNumberField(TEXT("loop_count"), LoopCountDouble);
	const int32 LoopCount = FMath::Max(1, static_cast<int32>(LoopCountDouble));

	if (!RuntimeSubsystem->PlayPedestrianAnimation(
			PedId,
			AnimationAssetPath,
			FName(*StartSection.TrimStartAndEnd()),
			static_cast<float>(PlayRate),
			LoopCount,
			Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetStringField(TEXT("animation_asset_path"), AnimationAssetPath.TrimStartAndEnd());
	return MakeSuccessResponse(TEXT("simAeroPedPlayAnimation"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedCommitCross(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, TEXT("ped_id is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector TargetWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("target_world_cm"), TEXT("target_enu_m"), WorldOriginCm, TargetWorldCm) &&
		!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, TargetWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, TEXT("target_world_cm or target_enu_m is required."));
	}

	bool bSnapToGround = true;
	TryReadBoolField(Payload, TEXT("snap_to_ground"), bSnapToGround);
	FString GroundSource;
	if (bSnapToGround)
	{
		SnapWorldPointToGround(GetWorld(), TargetWorldCm, nullptr, &GroundSource);
	}

	double SpeedCmPerSec = 0.0;
	Payload->TryGetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);
	if (!RuntimeSubsystem->CommitPedestrianCross(PedId, TargetWorldCm, static_cast<float>(SpeedCmPerSec), Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetNumberField(TEXT("speed_cm_per_sec"), SpeedCmPerSec);
	ResponsePayload->SetObjectField(TEXT("target_world_cm"), MakeVectorPayloadField(TargetWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("target_enu_m"), (TargetWorldCm - WorldOriginCm) / 100.0);
	if (!GroundSource.IsEmpty())
	{
		ResponsePayload->SetStringField(TEXT("ground_source"), GroundSource);
	}
	return MakeSuccessResponse(TEXT("simAeroPedCommitCross"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedStop(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedStop"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedStop"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedStop"), RequestId, MapId, TEXT("ped_id is required."));
	}
	if (!RuntimeSubsystem->StopPedestrian(PedId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedStop"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	return MakeSuccessResponse(TEXT("simAeroPedStop"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedSetVariant(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedSetVariant"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedSetVariant"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	FString VariantIdString;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	Payload->TryGetStringField(TEXT("variant_id"), VariantIdString);
	if (PedId.TrimStartAndEnd().IsEmpty() || VariantIdString.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedSetVariant"), RequestId, MapId, TEXT("ped_id and variant_id are required."));
	}

	if (!RuntimeSubsystem->SetPedestrianVariant(PedId, FName(*VariantIdString.TrimStartAndEnd()), Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSetVariant"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetStringField(TEXT("variant_id"), VariantIdString.TrimStartAndEnd());
	return MakeSuccessResponse(TEXT("simAeroPedSetVariant"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedRelease(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedRelease"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedRelease"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString PedId;
	Payload->TryGetStringField(TEXT("ped_id"), PedId);
	if (PedId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedRelease"), RequestId, MapId, TEXT("ped_id is required."));
	}
	if (!RuntimeSubsystem->ReleasePedestrian(PedId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedRelease"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("ped_id"), PedId);
	ResponsePayload->SetBoolField(TEXT("released"), true);
	return MakeSuccessResponse(TEXT("simAeroPedRelease"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandlePedSpawnCrowd(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawnCrowd"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawnCrowd"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawnCrowd"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FCrowdSpawnRequest SpawnRequest;
	FString GroupId;
	Payload->TryGetStringField(TEXT("group_id"), GroupId);
	SpawnRequest.GroupId = GroupId.TrimStartAndEnd().IsEmpty() ? FName(TEXT("crowd.default")) : FName(*GroupId.TrimStartAndEnd());
	double CountValue = 0.0;
	double SeedValue = 0.0;
	if (Payload->TryGetNumberField(TEXT("count"), CountValue))
	{
		SpawnRequest.Count = static_cast<int32>(CountValue);
	}
	if (Payload->TryGetNumberField(TEXT("seed"), SeedValue))
	{
		SpawnRequest.Seed = static_cast<int32>(SeedValue);
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("spawn_origin_world_cm"), TEXT("spawn_origin_enu_m"), WorldOriginCm, SpawnRequest.SpawnOrigin))
	{
		TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, SpawnRequest.SpawnOrigin);
	}

	bool bSnapToGround = true;
	TryReadBoolField(Payload, TEXT("snap_to_ground"), bSnapToGround);
	bool bUseProvidedGroundPoint = false;
	FString GroundSource;
	if (bSnapToGround)
	{
		bUseProvidedGroundPoint = SnapWorldPointToGround(GetWorld(), SpawnRequest.SpawnOrigin, nullptr, &GroundSource);
		if (!GroundSource.IsEmpty())
		{
			UE_LOG(
				LogAeroBridgeWorld,
				Log,
				TEXT("simAeroPedSpawnCrowd grounded: map_id='%s' group_id='%s' world='%s' source='%s'."),
				*(MapId.IsEmpty() ? CurrentMapId : MapId),
				*SpawnRequest.GroupId.ToString(),
				*SpawnRequest.SpawnOrigin.ToString(),
				*GroundSource);
		}
	}
	SpawnRequest.bUseProvidedGroundPoint = bUseProvidedGroundPoint;

	TryReadVectorField(Payload, TEXT("spawn_box_extent_cm"), SpawnRequest.SpawnBoxExtent);
	FString YawPolicyString;
	Payload->TryGetStringField(TEXT("yaw_policy"), YawPolicyString);
	if (!YawPolicyString.TrimStartAndEnd().IsEmpty())
	{
		SpawnRequest.YawPolicy = ParseCrowdYawPolicy(YawPolicyString);
	}
	Payload->TryGetNumberField(TEXT("fixed_yaw_deg"), SpawnRequest.FixedYawDeg);

	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	if (Settings != nullptr)
	{
		SpawnRequest.AppearancePool = Settings->DefaultCrowdAppearancePool.LoadSynchronous();
		SpawnRequest.RoleProfile = Settings->DefaultCrowdRoleProfile.LoadSynchronous();
	}

	FString AppearancePoolPath;
	Payload->TryGetStringField(TEXT("appearance_pool_path"), AppearancePoolPath);
	if (!AppearancePoolPath.TrimStartAndEnd().IsEmpty())
	{
		SpawnRequest.AppearancePool = Cast<UCrowdAppearancePool>(FSoftObjectPath(AppearancePoolPath.TrimStartAndEnd()).TryLoad());
	}

	FString RoleProfilePath;
	Payload->TryGetStringField(TEXT("role_profile_path"), RoleProfilePath);
	if (!RoleProfilePath.TrimStartAndEnd().IsEmpty())
	{
		SpawnRequest.RoleProfile = Cast<UCrowdRoleProfile>(FSoftObjectPath(RoleProfilePath.TrimStartAndEnd()).TryLoad());
	}

	FCrowdSpawnResult Result;
	if (!RuntimeSubsystem->SpawnCrowd(SpawnRequest, Result, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedSpawnCrowd"), RequestId, MapId, Error);
	}
	return MakeSuccessResponse(TEXT("simAeroPedSpawnCrowd"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, CrowdSpawnResultToJson(Result));
}

FString UAeroBridgeWorldSubsystem::HandlePedClearCrowd(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedClearCrowd"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedClearCrowd"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString GroupId;
	Payload->TryGetStringField(TEXT("group_id"), GroupId);
	if (GroupId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedClearCrowd"), RequestId, MapId, TEXT("group_id is required."));
	}

	const bool bCleared = RuntimeSubsystem->ClearCrowdGroup(FName(*GroupId.TrimStartAndEnd()), Error);
	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("group_id"), GroupId.TrimStartAndEnd());
	ResponsePayload->SetBoolField(TEXT("cleared"), bCleared);
	return bCleared
		? MakeSuccessResponse(TEXT("simAeroPedClearCrowd"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload)
		: MakeErrorResponse(TEXT("simAeroPedClearCrowd"), RequestId, MapId, Error);
}

FString UAeroBridgeWorldSubsystem::HandlePedRespawnCrowd(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroPedRespawnCrowd"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroPedRespawnCrowd"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString GroupId;
	Payload->TryGetStringField(TEXT("group_id"), GroupId);
	if (GroupId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroPedRespawnCrowd"), RequestId, MapId, TEXT("group_id is required."));
	}

	int32 Seed = 0;
	double SeedValue = 0.0;
	if (Payload->TryGetNumberField(TEXT("seed"), SeedValue))
	{
		Seed = static_cast<int32>(SeedValue);
	}
	FCrowdSpawnResult Result;
	if (!RuntimeSubsystem->RespawnCrowd(FName(*GroupId.TrimStartAndEnd()), Seed, Result, Error))
	{
		return MakeErrorResponse(TEXT("simAeroPedRespawnCrowd"), RequestId, MapId, Error);
	}
	return MakeSuccessResponse(TEXT("simAeroPedRespawnCrowd"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, CrowdSpawnResultToJson(Result));
}

FString UAeroBridgeWorldSubsystem::HandleSpawnAsset(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroSpawnAsset"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroSpawnAsset"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->SpawnAsset(Payload, Error) : nullptr;
	if (Result.IsValid() && AssetSubsystem != nullptr)
	{
		FString AssetId;
		FString LogicalAssetId;
		Result->TryGetStringField(TEXT("asset_id"), AssetId);
		Result->TryGetStringField(TEXT("logical_asset_id"), LogicalAssetId);

		FVector PositionEnuM = FVector::ZeroVector;
		TryReadVectorField(Result, TEXT("position_enu_m"), PositionEnuM);
		FVector WorldLocationCm = FVector::ZeroVector;
		if (!TryReadVectorField(Result, TEXT("position_world_cm"), WorldLocationCm))
		{
			const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
			WorldLocationCm = WorldOriginCm + PositionEnuM * 100.0;
		}
		FString GroundSource;
		Result->TryGetStringField(TEXT("ground_source"), GroundSource);
		const FAeroAssetInstanceState* Instance = AssetSubsystem->FindInstance(AssetId);
		const FString ActorName = (Instance != nullptr && Instance->Actor.IsValid()) ? Instance->Actor->GetName() : TEXT("<none>");
		UE_LOG(
			LogAeroBridgeWorld,
			Log,
			TEXT("simAeroSpawnAsset resolved: map_id='%s' asset_id='%s' logical_asset_id='%s' enu_m='%s' world_cm='%s' ground_source='%s' actor='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			*LogicalAssetId,
			*PositionEnuM.ToString(),
			*WorldLocationCm.ToString(),
			GroundSource.IsEmpty() ? TEXT("<none>") : *GroundSource,
			*ActorName);
	}
	else
	{
		FString AssetId;
		FString LogicalAssetId;
		Payload->TryGetStringField(TEXT("asset_id"), AssetId);
		Payload->TryGetStringField(TEXT("logical_asset_id"), LogicalAssetId);
		UE_LOG(
			LogAeroBridgeWorld,
			Warning,
			TEXT("simAeroSpawnAsset failed: map_id='%s' asset_id='%s' logical_asset_id='%s' error='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			*LogicalAssetId,
			AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : *Error);
	}
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroSpawnAsset"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroSpawnAsset"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleMoveAsset(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroMoveAsset"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroMoveAsset"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->MoveAsset(Payload, Error) : nullptr;
	if (Result.IsValid() && AssetSubsystem != nullptr)
	{
		FString AssetId;
		Result->TryGetStringField(TEXT("asset_id"), AssetId);

		FVector PositionEnuM = FVector::ZeroVector;
		TryReadVectorField(Result, TEXT("position_enu_m"), PositionEnuM);
		FVector WorldLocationCm = FVector::ZeroVector;
		if (!TryReadVectorField(Result, TEXT("position_world_cm"), WorldLocationCm))
		{
			const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
			WorldLocationCm = WorldOriginCm + PositionEnuM * 100.0;
		}
		FString GroundSource;
		Result->TryGetStringField(TEXT("ground_source"), GroundSource);
		const FAeroAssetInstanceState* Instance = AssetSubsystem->FindInstance(AssetId);
		const FString LogicalAssetId = Instance != nullptr ? Instance->LogicalAssetId : TEXT("<unknown>");
		const FString ActorName = (Instance != nullptr && Instance->Actor.IsValid()) ? Instance->Actor->GetName() : TEXT("<none>");
		UE_LOG(
			LogAeroBridgeWorld,
			Log,
			TEXT("simAeroMoveAsset resolved: map_id='%s' asset_id='%s' logical_asset_id='%s' enu_m='%s' world_cm='%s' ground_source='%s' actor='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			*LogicalAssetId,
			*PositionEnuM.ToString(),
			*WorldLocationCm.ToString(),
			GroundSource.IsEmpty() ? TEXT("<none>") : *GroundSource,
			*ActorName);
	}
	else
	{
		FString AssetId;
		Payload->TryGetStringField(TEXT("asset_id"), AssetId);
		UE_LOG(
			LogAeroBridgeWorld,
			Warning,
			TEXT("simAeroMoveAsset failed: map_id='%s' asset_id='%s' error='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : *Error);
	}
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroMoveAsset"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroMoveAsset"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleRemoveAsset(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroRemoveAsset"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroRemoveAsset"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	FString AssetId;
	Payload->TryGetStringField(TEXT("asset_id"), AssetId);
	const FAeroAssetInstanceState* ExistingInstance = AssetSubsystem != nullptr ? AssetSubsystem->FindInstance(AssetId) : nullptr;
	const FString ExistingLogicalAssetId = ExistingInstance != nullptr ? ExistingInstance->LogicalAssetId : TEXT("<unknown>");
	const FString ExistingActorName = (ExistingInstance != nullptr && ExistingInstance->Actor.IsValid()) ? ExistingInstance->Actor->GetName() : TEXT("<none>");
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->RemoveAsset(Payload, Error) : nullptr;
	if (Result.IsValid())
	{
		UE_LOG(
			LogAeroBridgeWorld,
			Log,
			TEXT("simAeroRemoveAsset resolved: map_id='%s' asset_id='%s' logical_asset_id='%s' actor='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			*ExistingLogicalAssetId,
			*ExistingActorName);
	}
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroRemoveAsset"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroRemoveAsset"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleCaptureWorldCamera(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	if (AssetSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId, TEXT("AeroAssetPlacement subsystem unavailable."));
	}

	FString AssetId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), AssetId) || AssetId.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId, TEXT("asset_id is required."));
	}
	AssetId = AssetId.TrimStartAndEnd();

	const FAeroAssetInstanceState* Instance = AssetSubsystem->FindInstance(AssetId);
	if (Instance == nullptr)
	{
		return MakeErrorResponse(
			TEXT("simAeroCaptureWorldCamera"),
			RequestId,
			MapId,
			FString::Printf(TEXT("Unknown asset_id: %s"), *AssetId));
	}
	if (!Instance->Actor.IsValid())
	{
		return MakeErrorResponse(
			TEXT("simAeroCaptureWorldCamera"),
			RequestId,
			MapId,
			FString::Printf(TEXT("asset_id '%s' has no live actor."), *AssetId));
	}

	AAeroFixedWorldCaptureCamera* CaptureActor = Cast<AAeroFixedWorldCaptureCamera>(Instance->Actor.Get());
	if (!IsValid(CaptureActor))
	{
		return MakeErrorResponse(
			TEXT("simAeroCaptureWorldCamera"),
			RequestId,
			MapId,
			FString::Printf(TEXT("asset_id '%s' is not a fixed world capture camera."), *AssetId));
	}

	double WidthValue = 1280.0;
	double HeightValue = 720.0;
	double FovDegreesValue = 70.0;
	Payload->TryGetNumberField(TEXT("width"), WidthValue);
	Payload->TryGetNumberField(TEXT("height"), HeightValue);
	Payload->TryGetNumberField(TEXT("fov_degrees"), FovDegreesValue);
	const int32 Width = FMath::Max(1, FMath::RoundToInt(WidthValue));
	const int32 Height = FMath::Max(1, FMath::RoundToInt(HeightValue));

	FString Modality;
	Payload->TryGetStringField(TEXT("modality"), Modality);
	Modality = Modality.TrimStartAndEnd().ToLower();
	if (Modality.IsEmpty())
	{
		Modality = TEXT("rgb");
	}
	if (!Modality.Equals(TEXT("rgb"), ESearchCase::IgnoreCase)
		&& !Modality.Equals(TEXT("depth"), ESearchCase::IgnoreCase)
		&& !Modality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		return MakeErrorResponse(
			TEXT("simAeroCaptureWorldCamera"),
			RequestId,
			MapId,
			FString::Printf(TEXT("Unsupported capture modality '%s'. Expected rgb, depth, or seg."), *Modality));
	}
	const FString Extension = Modality.Equals(TEXT("depth"), ESearchCase::IgnoreCase) ? TEXT("npy") : TEXT("png");

	FString OutputPath;
	Payload->TryGetStringField(TEXT("output_path"), OutputPath);
	if (OutputPath.IsEmpty())
	{
		FString OutputDir;
		FString FileName;
		Payload->TryGetStringField(TEXT("output_dir"), OutputDir);
		Payload->TryGetStringField(TEXT("file_name"), FileName);
		if (OutputDir.IsEmpty())
		{
			OutputDir = TEXT("Saved/AeroBridge/fixed_world_camera");
		}
		if (FileName.IsEmpty())
		{
			FileName = FString::Printf(TEXT("%s.%s"), *AssetId, *Extension);
		}
		else if (!FileName.EndsWith(FString::Printf(TEXT(".%s"), *Extension), ESearchCase::IgnoreCase))
		{
			FileName = FPaths::SetExtension(FileName, Extension);
		}
		OutputPath = FPaths::Combine(ResolveRelativePath(OutputDir), FileName);
	}
	else
	{
		OutputPath = ResolveRelativePath(OutputPath);
		if (!OutputPath.EndsWith(FString::Printf(TEXT(".%s"), *Extension), ESearchCase::IgnoreCase))
		{
			OutputPath = FPaths::SetExtension(OutputPath, Extension);
		}
	}

	FString SemanticRulesPath;
	FString SemanticAuditPath;
	if (Modality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		Payload->TryGetStringField(TEXT("semantic_rules_path"), SemanticRulesPath);
		Payload->TryGetStringField(TEXT("semantic_audit_path"), SemanticAuditPath);
		if (SemanticRulesPath.TrimStartAndEnd().IsEmpty())
		{
			SemanticRulesPath = AeroSemanticStencil::DefaultRulesPath();
		}
		else
		{
			SemanticRulesPath = ResolveRelativePath(SemanticRulesPath);
		}
		if (!SemanticAuditPath.TrimStartAndEnd().IsEmpty())
		{
			SemanticAuditPath = ResolveRelativePath(SemanticAuditPath);
		}
	}

	FAeroFixedWorldCaptureStats CaptureStats;
	if (!CaptureActor->CaptureToDisk(
		Modality,
		OutputPath,
		Width,
		Height,
		static_cast<float>(FovDegreesValue),
		Error,
		CaptureStats,
		SemanticRulesPath,
		SemanticAuditPath))
	{
		UE_LOG(
			LogAeroBridgeWorld,
			Warning,
			TEXT("simAeroCaptureWorldCamera failed: map_id='%s' asset_id='%s' actor='%s' modality='%s' output='%s' error='%s'."),
			*(MapId.IsEmpty() ? CurrentMapId : MapId),
			*AssetId,
			*CaptureActor->GetName(),
			*Modality,
			*OutputPath,
			*Error);
		return MakeErrorResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId, Error);
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	const FVector ActorWorldCm = CaptureActor->GetActorLocation();
	const FVector PositionEnuM = (ActorWorldCm - WorldOriginCm) / 100.0;
	const FRotator RotationDeg = CaptureActor->GetActorRotation();

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("asset_id"), AssetId);
	ResponsePayload->SetStringField(TEXT("logical_asset_id"), Instance->LogicalAssetId);
	ResponsePayload->SetStringField(TEXT("actor_name"), CaptureActor->GetName());
	ResponsePayload->SetStringField(TEXT("modality"), Modality);
	ResponsePayload->SetStringField(TEXT("output_format"), CaptureStats.OutputFormat);
	ResponsePayload->SetStringField(TEXT("output_path"), OutputPath);
	ResponsePayload->SetNumberField(TEXT("width"), CaptureStats.CapturedWidth);
	ResponsePayload->SetNumberField(TEXT("height"), CaptureStats.CapturedHeight);
	if (Modality.Equals(TEXT("depth"), ESearchCase::IgnoreCase))
	{
		ResponsePayload->SetBoolField(TEXT("depth_unit_m"), CaptureStats.bDepthUnitMeters);
		ResponsePayload->SetNumberField(TEXT("depth_min_m"), CaptureStats.DepthMinM);
		ResponsePayload->SetNumberField(TEXT("depth_max_m"), CaptureStats.DepthMaxM);
		ResponsePayload->SetNumberField(TEXT("depth_valid_count"), CaptureStats.DepthValidCount);
		ResponsePayload->SetNumberField(TEXT("depth_invalid_count"), CaptureStats.DepthInvalidCount);
	}
	if (Modality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		ResponsePayload->SetStringField(TEXT("segmentation_kind"), CaptureStats.SegmentationKind);
		ResponsePayload->SetStringField(TEXT("semantic_rules_path"), CaptureStats.SemanticRulesPath);
		ResponsePayload->SetStringField(TEXT("semantic_audit_path"), CaptureStats.SemanticAuditPath);
		ResponsePayload->SetObjectField(TEXT("semantic_class_by_id"), MakeClassIdToNamePayload(CaptureStats.SemanticClassById));
		ResponsePayload->SetObjectField(TEXT("class_histogram"), MakeClassIdHistogramPayload(CaptureStats.SemanticClassHistogram));
		ResponsePayload->SetNumberField(TEXT("ignore_pixel_count"), CaptureStats.IgnorePixelCount);
		ResponsePayload->SetNumberField(TEXT("invalid_semantic_class_id_pixel_count"), CaptureStats.SemanticInvalidClassIdPixelCount);
		ResponsePayload->SetNumberField(TEXT("unknown_semantic_color_pixel_count"), CaptureStats.SemanticUnknownColorPixelCount);
		ResponsePayload->SetNumberField(TEXT("semantic_assigned_component_count"), CaptureStats.SemanticAssignedComponentCount);
	}
	ResponsePayload->SetObjectField(TEXT("position_world_cm"), MakeVectorPayloadField(ActorWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("position_enu_m"), PositionEnuM);
	SetRotatorObjectField(ResponsePayload, TEXT("rotation_deg"), RotationDeg);

	UE_LOG(
		LogAeroBridgeWorld,
		Log,
		TEXT("simAeroCaptureWorldCamera resolved: map_id='%s' asset_id='%s' logical_asset_id='%s' actor='%s' modality='%s' output_format='%s' output='%s' enu_m='%s' rotation='%s'."),
		*(MapId.IsEmpty() ? CurrentMapId : MapId),
		*AssetId,
		*Instance->LogicalAssetId,
		*CaptureActor->GetName(),
		*Modality,
		*CaptureStats.OutputFormat,
		*OutputPath,
		*PositionEnuM.ToString(),
		*RotationDeg.ToString());

	return MakeSuccessResponse(TEXT("simAeroCaptureWorldCamera"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleSemanticStencilAudit(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("aero.semantic_stencil_audit_json"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("aero.semantic_stencil_audit_json"), RequestId, MapId, Error);
	}

	FString RulesPath;
	FString AuditPath;
	bool bAssign = false;
	Payload->TryGetStringField(TEXT("semantic_rules_path"), RulesPath);
	Payload->TryGetStringField(TEXT("rules_path"), RulesPath);
	Payload->TryGetStringField(TEXT("semantic_audit_path"), AuditPath);
	Payload->TryGetStringField(TEXT("audit_path"), AuditPath);
	Payload->TryGetBoolField(TEXT("assign"), bAssign);
	RulesPath = RulesPath.TrimStartAndEnd().IsEmpty() ? AeroSemanticStencil::DefaultRulesPath() : ResolveRelativePath(RulesPath);
	if (!AuditPath.TrimStartAndEnd().IsEmpty())
	{
		AuditPath = ResolveRelativePath(AuditPath);
	}

	TSet<const AActor*> IgnoredActors;
	FAeroSemanticStencilAudit Audit;
	if (!AeroSemanticStencil::AuditAndAssign(GetWorld(), RulesPath, bAssign, IgnoredActors, Audit, Error))
	{
		return MakeErrorResponse(TEXT("aero.semantic_stencil_audit_json"), RequestId, MapId, Error);
	}
	if (!AuditPath.IsEmpty() && !AeroSemanticStencil::SaveAuditJson(Audit, AuditPath, Error))
	{
		return MakeErrorResponse(TEXT("aero.semantic_stencil_audit_json"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = AeroSemanticStencil::AuditToJson(Audit, false);
	ResponsePayload->SetStringField(TEXT("audit_path"), AuditPath);
	return MakeSuccessResponse(TEXT("aero.semantic_stencil_audit_json"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleReserveOccupancy(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroReserveOccupancy"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroReserveOccupancy"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->ReserveOccupancy(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroReserveOccupancy"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroReserveOccupancy"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleReleaseOccupancy(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroReleaseOccupancy"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroReleaseOccupancy"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->ReleaseOccupancy(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroReleaseOccupancy"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroReleaseOccupancy"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleQueryNearest(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroQueryNearest"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroQueryNearest"), RequestId, MapId, Error);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	TSharedPtr<FJsonObject> Result = AssetSubsystem != nullptr ? AssetSubsystem->QueryNearest(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroQueryNearest"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroQueryNearest"), RequestId, MapId, AssetSubsystem == nullptr ? TEXT("AeroAssetPlacement subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleQueryPedPath(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroQueryPedPath"), RequestId, MapId, Error);
	}

	UAeroPedNavSemanticSubsystem* PedSubsystem = GetWorld()->GetSubsystem<UAeroPedNavSemanticSubsystem>();
	TSharedPtr<FJsonObject> Result = PedSubsystem != nullptr ? PedSubsystem->QueryPedPath(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroQueryPedPath"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroQueryPedPath"), RequestId, MapId, PedSubsystem == nullptr ? TEXT("AeroPedNavSemantic subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleProjectGround(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroProjectGround"), RequestId, MapId, Error);
	}

	UAeroPedNavSemanticSubsystem* PedSubsystem = GetWorld()->GetSubsystem<UAeroPedNavSemanticSubsystem>();
	TSharedPtr<FJsonObject> Result = PedSubsystem != nullptr ? PedSubsystem->ProjectGround(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroProjectGround"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroProjectGround"), RequestId, MapId, PedSubsystem == nullptr ? TEXT("AeroPedNavSemantic subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleQueryPedAnchor(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroQueryPedAnchor"), RequestId, MapId, Error);
	}

	UAeroPedNavSemanticSubsystem* PedSubsystem = GetWorld()->GetSubsystem<UAeroPedNavSemanticSubsystem>();
	TSharedPtr<FJsonObject> Result = PedSubsystem != nullptr ? PedSubsystem->QueryPedAnchor(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroQueryPedAnchor"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroQueryPedAnchor"), RequestId, MapId, PedSubsystem == nullptr ? TEXT("AeroPedNavSemantic subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleApplyWeather(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroApplyWeather"), RequestId, MapId, Error);
	}

	UAeroWeatherRenderSubsystem* WeatherSubsystem = GetWorld()->GetSubsystem<UAeroWeatherRenderSubsystem>();
	TSharedPtr<FJsonObject> Result = WeatherSubsystem != nullptr ? WeatherSubsystem->ApplyWeather(Payload, Error) : nullptr;
	return Result.IsValid()
		? MakeSuccessResponse(TEXT("simAeroApplyWeather"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, Result)
		: MakeErrorResponse(TEXT("simAeroApplyWeather"), RequestId, MapId, WeatherSubsystem == nullptr ? TEXT("AeroWeatherRender subsystem unavailable.") : Error);
}

FString UAeroBridgeWorldSubsystem::HandleCreateRuntimeMultirotor(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString VehicleName;
	Payload->TryGetStringField(TEXT("vehicle_name"), VehicleName);
	if (VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, TEXT("vehicle_name is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector SpawnWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, SpawnWorldCm) &&
		!TryResolveWorldPointFromPayload(Payload, TEXT("spawn_world_cm"), TEXT("spawn_enu_m"), WorldOriginCm, SpawnWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, TEXT("position_world_cm or position_enu_m is required."));
	}

	FRotator SpawnRotation = FRotator::ZeroRotator;
	if (!TryReadRotatorField(Payload, TEXT("rotation_deg"), SpawnRotation) &&
		!TryReadRotatorField(Payload, TEXT("world_rotation_deg"), SpawnRotation))
	{
		TryReadRotatorFieldFromObject(Payload, SpawnRotation);
	}

	if (!RuntimeSubsystem->CreateRuntimeMultirotor(VehicleName.TrimStartAndEnd(), SpawnWorldCm, SpawnRotation, Error))
	{
		return MakeErrorResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId, Error);
	}

	FVector AuthoritativeWorldCm = SpawnWorldCm;
	FRotator AuthoritativeRotation = SpawnRotation;
	FString PoseError;
	RuntimeSubsystem->GetVehiclePose(VehicleName.TrimStartAndEnd(), AuthoritativeWorldCm, AuthoritativeRotation, PoseError);

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("vehicle_name"), VehicleName.TrimStartAndEnd());
	ResponsePayload->SetBoolField(TEXT("created"), true);
	ResponsePayload->SetObjectField(TEXT("position_world_cm"), MakeVectorPayloadField(AuthoritativeWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("position_enu_m"), (AuthoritativeWorldCm - WorldOriginCm) / 100.0);
	SetRotatorObjectField(ResponsePayload, TEXT("rotation_deg"), AuthoritativeRotation);
	return MakeSuccessResponse(TEXT("simAeroCreateRuntimeMultirotor"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleMoveRuntimeMultirotor(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, Error);
	}
	if (!MapId.IsEmpty() && MapId != CurrentMapId && !LoadContextByMapId(MapId, Error))
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString VehicleName;
	Payload->TryGetStringField(TEXT("vehicle_name"), VehicleName);
	if (VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, TEXT("vehicle_name is required."));
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	FVector TargetWorldCm = FVector::ZeroVector;
	if (!TryResolveWorldPointFromPayload(Payload, TEXT("target_world_cm"), TEXT("target_enu_m"), WorldOriginCm, TargetWorldCm) &&
		!TryResolveWorldPointFromPayload(Payload, TEXT("position_world_cm"), TEXT("position_enu_m"), WorldOriginCm, TargetWorldCm))
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, TEXT("target_world_cm or target_enu_m is required."));
	}

	double VelocityMps = 5.0;
	Payload->TryGetNumberField(TEXT("velocity_mps"), VelocityMps);
	if (!RuntimeSubsystem->MoveMultirotorToPosition(VehicleName.TrimStartAndEnd(), TargetWorldCm, static_cast<float>(VelocityMps), Error))
	{
		return MakeErrorResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("vehicle_name"), VehicleName.TrimStartAndEnd());
	ResponsePayload->SetStringField(TEXT("state"), TEXT("running"));
	ResponsePayload->SetNumberField(TEXT("velocity_mps"), VelocityMps);
	ResponsePayload->SetObjectField(TEXT("target_world_cm"), MakeVectorPayloadField(TargetWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("target_enu_m"), (TargetWorldCm - WorldOriginCm) / 100.0);
	return MakeSuccessResponse(TEXT("simAeroMoveRuntimeMultirotor"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleGetRuntimeMultirotorStatus(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeMultirotorStatus"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeMultirotorStatus"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString VehicleName;
	Payload->TryGetStringField(TEXT("vehicle_name"), VehicleName);
	if (VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeMultirotorStatus"), RequestId, MapId, TEXT("vehicle_name is required."));
	}

	FAeroRuntimeMoveStatus Status;
	if (!RuntimeSubsystem->GetMultirotorMoveStatus(VehicleName.TrimStartAndEnd(), Status, Error))
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeMultirotorStatus"), RequestId, MapId, Error);
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("vehicle_name"), VehicleName.TrimStartAndEnd());
	ResponsePayload->SetStringField(TEXT("state"), ToRuntimeMoveStateJsonString(Status.State));
	ResponsePayload->SetStringField(TEXT("message"), Status.Message);
	ResponsePayload->SetNumberField(TEXT("velocity_mps"), Status.VelocityMps);
	ResponsePayload->SetObjectField(TEXT("target_world_cm"), MakeVectorPayloadField(Status.TargetWorldCm));
	SetVectorArrayField(ResponsePayload, TEXT("target_enu_m"), (Status.TargetWorldCm - WorldOriginCm) / 100.0);

	FVector CurrentWorldCm = FVector::ZeroVector;
	FRotator CurrentRotation = FRotator::ZeroRotator;
	FString PoseError;
	if (RuntimeSubsystem->GetVehiclePose(VehicleName.TrimStartAndEnd(), CurrentWorldCm, CurrentRotation, PoseError))
	{
		ResponsePayload->SetObjectField(TEXT("current_world_cm"), MakeVectorPayloadField(CurrentWorldCm));
		SetVectorArrayField(ResponsePayload, TEXT("current_enu_m"), (CurrentWorldCm - WorldOriginCm) / 100.0);
		SetRotatorObjectField(ResponsePayload, TEXT("current_rotation_deg"), CurrentRotation);
	}
	return MakeSuccessResponse(TEXT("simAeroGetRuntimeMultirotorStatus"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleRemoveRuntimeVehicle(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroRemoveRuntimeVehicle"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroRemoveRuntimeVehicle"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString VehicleName;
	Payload->TryGetStringField(TEXT("vehicle_name"), VehicleName);
	if (VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroRemoveRuntimeVehicle"), RequestId, MapId, TEXT("vehicle_name is required."));
	}

	if (!RuntimeSubsystem->RemoveRuntimeVehicle(VehicleName.TrimStartAndEnd(), Error))
	{
		return MakeErrorResponse(TEXT("simAeroRemoveRuntimeVehicle"), RequestId, MapId, Error);
	}

	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("vehicle_name"), VehicleName.TrimStartAndEnd());
	ResponsePayload->SetBoolField(TEXT("removed"), true);
	return MakeSuccessResponse(TEXT("simAeroRemoveRuntimeVehicle"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}

FString UAeroBridgeWorldSubsystem::HandleGetRuntimeVehiclePose(const FString& RequestJson)
{
	FString RequestId;
	FString MapId;
	FString Error;
	TSharedPtr<FJsonObject> Payload = ParseRequestEnvelope(RequestJson, RequestId, MapId, Error);
	if (!Payload.IsValid())
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeVehiclePose"), RequestId, MapId, Error);
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = ResolveRuntimeOrchestrationSubsystem(GetWorld());
	if (RuntimeSubsystem == nullptr)
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeVehiclePose"), RequestId, MapId, TEXT("AeroRuntimeOrchestrationSubsystem unavailable."));
	}

	FString VehicleName;
	Payload->TryGetStringField(TEXT("vehicle_name"), VehicleName);
	if (VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeVehiclePose"), RequestId, MapId, TEXT("vehicle_name is required."));
	}

	FVector WorldLocationCm = FVector::ZeroVector;
	FRotator WorldRotation = FRotator::ZeroRotator;
	if (!RuntimeSubsystem->GetVehiclePose(VehicleName.TrimStartAndEnd(), WorldLocationCm, WorldRotation, Error))
	{
		return MakeErrorResponse(TEXT("simAeroGetRuntimeVehiclePose"), RequestId, MapId, Error);
	}

	const FVector WorldOriginCm = ReadWorldOriginCm(CurrentMapContext);
	TSharedPtr<FJsonObject> ResponsePayload = MakeShared<FJsonObject>();
	ResponsePayload->SetStringField(TEXT("vehicle_name"), VehicleName.TrimStartAndEnd());
	ResponsePayload->SetObjectField(TEXT("position_world_cm"), MakeVectorPayloadField(WorldLocationCm));
	SetVectorArrayField(ResponsePayload, TEXT("position_enu_m"), (WorldLocationCm - WorldOriginCm) / 100.0);
	SetRotatorObjectField(ResponsePayload, TEXT("rotation_deg"), WorldRotation);
	return MakeSuccessResponse(TEXT("simAeroGetRuntimeVehiclePose"), RequestId, MapId.IsEmpty() ? CurrentMapId : MapId, ResponsePayload);
}
