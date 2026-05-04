#include "SumoRoadNetworkActor.h"

#include "Components/SceneComponent.h"
#include "Components/SplineComponent.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "SumoImporterLog.h"

namespace
{
bool IsBetterLaneSampleOnActorCandidate(
	const float CandidateDistance2DCm,
	const float CandidateAbsZDeltaCm,
	const bool bHasBest,
	const float BestDistance2DCm,
	const float BestAbsZDeltaCm)
{
	if (!bHasBest)
	{
		return true;
	}

	if (CandidateDistance2DCm + KINDA_SMALL_NUMBER < BestDistance2DCm)
	{
		return true;
	}

	return FMath::IsNearlyEqual(CandidateDistance2DCm, BestDistance2DCm, 0.1f) &&
		CandidateAbsZDeltaCm + KINDA_SMALL_NUMBER < BestAbsZDeltaCm;
}

FString EscapeCsvField(const FString& Value)
{
	if (!Value.Contains(TEXT(",")) && !Value.Contains(TEXT("\"")) && !Value.Contains(TEXT("\n")) && !Value.Contains(TEXT("\r")))
	{
		return Value;
	}

	FString Escaped = Value;
	Escaped.ReplaceInline(TEXT("\""), TEXT("\"\""));
	return FString::Printf(TEXT("\"%s\""), *Escaped);
}

FString ResolveOutputPath(const FString& Path)
{
	FString ResolvedPath = Path;
	if (FPaths::IsRelative(ResolvedPath))
	{
		ResolvedPath = FPaths::Combine(FPaths::ProjectSavedDir(), ResolvedPath);
	}
	return FPaths::ConvertRelativePathToFull(ResolvedPath);
}

bool EnsureOutputDirectory(const FString& FilePath)
{
	const FString Directory = FPaths::GetPath(FilePath);
	return Directory.IsEmpty() || IFileManager::Get().MakeDirectory(*Directory, true);
}

float SanitizeRoadWeight(float RoadWeight)
{
	return (RoadWeight > KINDA_SMALL_NUMBER) ? RoadWeight : 1.0f;
}
}

ASumoRoadNetworkActor::ASumoRoadNetworkActor()
{
	PrimaryActorTick.bCanEverTick = false;

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SceneRoot->SetMobility(EComponentMobility::Movable);
	SetRootComponent(SceneRoot);
}

void ASumoRoadNetworkActor::ResetNetwork()
{
	for (USplineComponent* Spline : LaneSplineComponents)
	{
		if (IsValid(Spline))
		{
			Spline->DestroyComponent();
		}
	}

	for (USplineComponent* JunctionSpline : JunctionDebugSplines)
	{
		if (IsValid(JunctionSpline))
		{
			JunctionSpline->DestroyComponent();
		}
	}

	LaneSplineComponents.Reset();
	JunctionDebugSplines.Reset();
	RuntimeLanes.Reset();
	RawConnections.Reset();
	LaneIdToRuntimeIndex.Reset();
	EdgeIdToRuntimeIndices.Reset();
	EdgeLaneToRuntimeIndex.Reset();
	SuccessorConnectionsByLaneKey.Reset();
}

bool ASumoRoadNetworkActor::AddLane(const FSumoLaneData& LaneData, const FString& EdgeType, float RoadWeight, const TArray<FVector>& LanePointsWorld)
{
	if (LanePointsWorld.Num() < 2)
	{
		return false;
	}

	const FName ComponentName(*FString::Printf(TEXT("LaneSpline_%d"), LaneSplineComponents.Num()));
	USplineComponent* SplineComponent = NewObject<USplineComponent>(this, ComponentName, RF_Transactional);
	if (!IsValid(SplineComponent))
	{
		return false;
	}

	SplineComponent->SetupAttachment(GetRootComponent());
	SplineComponent->SetMobility(EComponentMobility::Movable);
	AddInstanceComponent(SplineComponent);
	SplineComponent->RegisterComponent();

	SplineComponent->ClearSplinePoints(false);
	for (const FVector& PointWorld : LanePointsWorld)
	{
		SplineComponent->AddSplinePoint(PointWorld, ESplineCoordinateSpace::World, false);
	}
	SplineComponent->SetClosedLoop(false, false);
	SplineComponent->UpdateSpline();

	FSumoLaneRuntimeData RuntimeData;
	RuntimeData.LaneHandle.LaneId = LaneData.LaneId;
	RuntimeData.LaneHandle.EdgeId = LaneData.EdgeId;
	RuntimeData.LaneHandle.LaneIndex = LaneData.LaneIndex;
	RuntimeData.LaneHandle.SpeedMps = LaneData.SpeedMps;
	RuntimeData.LaneHandle.EdgeType = EdgeType;
	RuntimeData.LaneHandle.RoadWeight = SanitizeRoadWeight(RoadWeight);
	RuntimeData.LaneHandle.LengthM = LaneData.LengthM > 0.0f
		? LaneData.LengthM
		: SplineComponent->GetSplineLength() * 0.01f;
	RuntimeData.SplineComponent = SplineComponent;

	const int32 RuntimeIndex = RuntimeLanes.Add(RuntimeData);
	LaneSplineComponents.Add(SplineComponent);
	LaneIdToRuntimeIndex.Add(RuntimeData.LaneHandle.LaneId, RuntimeIndex);
	EdgeIdToRuntimeIndices.FindOrAdd(RuntimeData.LaneHandle.EdgeId).Add(RuntimeIndex);
	EdgeLaneToRuntimeIndex.Add(MakeEdgeLaneKey(RuntimeData.LaneHandle.EdgeId, RuntimeData.LaneHandle.LaneIndex), RuntimeIndex);
	return true;
}

void ASumoRoadNetworkActor::AddJunctionDebugPolygon(const FSumoJunctionData& JunctionData, const TArray<FVector>& PolygonPointsWorld)
{
	if (PolygonPointsWorld.Num() < 2)
	{
		return;
	}

	FString SanitizedJunctionId = JunctionData.JunctionId;
	SanitizedJunctionId.ReplaceInline(TEXT(":"), TEXT("_"));
	SanitizedJunctionId.ReplaceInline(TEXT("#"), TEXT("_"));
	SanitizedJunctionId.ReplaceInline(TEXT("."), TEXT("_"));
	SanitizedJunctionId.ReplaceInline(TEXT("/"), TEXT("_"));
	SanitizedJunctionId.ReplaceInline(TEXT("\\"), TEXT("_"));
	const FName ComponentName(*FString::Printf(TEXT("JunctionSpline_%s"), *SanitizedJunctionId));
	USplineComponent* JunctionSpline = NewObject<USplineComponent>(this, ComponentName, RF_Transactional);
	if (!IsValid(JunctionSpline))
	{
		return;
	}

	JunctionSpline->SetupAttachment(GetRootComponent());
	JunctionSpline->SetMobility(EComponentMobility::Movable);
	AddInstanceComponent(JunctionSpline);
	JunctionSpline->RegisterComponent();

	JunctionSpline->ClearSplinePoints(false);
	for (const FVector& PointWorld : PolygonPointsWorld)
	{
		JunctionSpline->AddSplinePoint(PointWorld, ESplineCoordinateSpace::World, false);
	}

	const bool bCloseLoop = PolygonPointsWorld.Num() >= 3;
	JunctionSpline->SetClosedLoop(bCloseLoop, false);
	JunctionSpline->UpdateSpline();

	JunctionDebugSplines.Add(JunctionSpline);
}

void ASumoRoadNetworkActor::SetConnections(const TArray<FSumoConnectionData>& InConnections)
{
	RawConnections = InConnections;
	SuccessorConnectionsByLaneKey.Reset();

	for (const FSumoConnectionData& Connection : RawConnections)
	{
		const FString LaneKey = MakeEdgeLaneKey(Connection.FromEdge, Connection.FromLane);
		if (EdgeLaneToRuntimeIndex.Contains(LaneKey))
		{
			SuccessorConnectionsByLaneKey.FindOrAdd(LaneKey).Add(Connection);
		}
	}
}

bool ASumoRoadNetworkActor::FindLaneById(const FString& LaneId, FSumoLaneHandle& OutLane) const
{
	const int32* RuntimeIndex = LaneIdToRuntimeIndex.Find(LaneId);
	if (RuntimeIndex == nullptr || !RuntimeLanes.IsValidIndex(*RuntimeIndex))
	{
		return false;
	}

	OutLane = RuntimeLanes[*RuntimeIndex].LaneHandle;
	return true;
}

void ASumoRoadNetworkActor::FindLanesByEdge(const FString& EdgeId, TArray<FSumoLaneHandle>& OutLanes) const
{
	OutLanes.Reset();

	const TArray<int32>* RuntimeIndices = EdgeIdToRuntimeIndices.Find(EdgeId);
	if (RuntimeIndices == nullptr)
	{
		return;
	}

	for (const int32 RuntimeIndex : *RuntimeIndices)
	{
		if (RuntimeLanes.IsValidIndex(RuntimeIndex))
		{
			OutLanes.Add(RuntimeLanes[RuntimeIndex].LaneHandle);
		}
	}
}

bool ASumoRoadNetworkActor::SampleTransformByEdgeLane(
	const FString& EdgeId,
	int32 LaneIndex,
	float SInMeters,
	FTransform& OutTransform,
	bool& bWasClamped) const
{
	bWasClamped = false;

	int32 RuntimeIndex = INDEX_NONE;
	if (!ResolveLaneRuntimeIndex(EdgeId, LaneIndex, RuntimeIndex))
	{
		return false;
	}

	const FSumoLaneRuntimeData& RuntimeLane = RuntimeLanes[RuntimeIndex];
	if (!IsValid(RuntimeLane.SplineComponent))
	{
		return false;
	}

	const float SplineLengthCm = RuntimeLane.SplineComponent->GetSplineLength();
	const float RequestedDistanceCm = SInMeters * 100.0f;
	const float ClampedDistanceCm = FMath::Clamp(RequestedDistanceCm, 0.0f, SplineLengthCm);
	bWasClamped = !FMath::IsNearlyEqual(RequestedDistanceCm, ClampedDistanceCm, 0.1f);

	const FVector Location = RuntimeLane.SplineComponent->GetLocationAtDistanceAlongSpline(ClampedDistanceCm, ESplineCoordinateSpace::World);
	const FVector Direction = RuntimeLane.SplineComponent->GetDirectionAtDistanceAlongSpline(ClampedDistanceCm, ESplineCoordinateSpace::World);
	OutTransform = FTransform(Direction.Rotation(), Location, FVector::OneVector);
	return true;
}

void ASumoRoadNetworkActor::GetSuccessors(const FString& FromEdge, int32 FromLane, TArray<FSumoConnectionData>& OutSuccessors) const
{
	OutSuccessors.Reset();
	const FString LaneKey = MakeEdgeLaneKey(FromEdge, FromLane);
	if (const TArray<FSumoConnectionData>* FoundConnections = SuccessorConnectionsByLaneKey.Find(LaneKey))
	{
		OutSuccessors = *FoundConnections;
	}
}

bool ASumoRoadNetworkActor::FindNearestLaneSample(const FVector& QueryWorldCm, FSumoNearestLaneSample& OutSample) const
{
	OutSample = FSumoNearestLaneSample();

	bool bFoundSample = false;
	float BestDistance2DCm = 0.0f;
	float BestAbsZDeltaCm = 0.0f;

	for (const FSumoLaneRuntimeData& RuntimeLane : RuntimeLanes)
	{
		if (!IsValid(RuntimeLane.SplineComponent))
		{
			continue;
		}

		const float InputKey = RuntimeLane.SplineComponent->FindInputKeyClosestToWorldLocation(QueryWorldCm);
		const float DistanceAlongSplineCm = RuntimeLane.SplineComponent->GetDistanceAlongSplineAtSplineInputKey(InputKey);
		const FTransform CandidateTransform = RuntimeLane.SplineComponent->GetTransformAtDistanceAlongSpline(
			DistanceAlongSplineCm,
			ESplineCoordinateSpace::World,
			true);
		const FVector CandidateLocation = CandidateTransform.GetLocation();
		const float CandidateDistance2DCm = FVector::Dist2D(CandidateLocation, QueryWorldCm);
		const float CandidateAbsZDeltaCm = FMath::Abs(CandidateLocation.Z - QueryWorldCm.Z);
		if (!IsBetterLaneSampleOnActorCandidate(
				CandidateDistance2DCm,
				CandidateAbsZDeltaCm,
				bFoundSample,
				BestDistance2DCm,
				BestAbsZDeltaCm))
		{
			continue;
		}

		bFoundSample = true;
		BestDistance2DCm = CandidateDistance2DCm;
		BestAbsZDeltaCm = CandidateAbsZDeltaCm;
		OutSample.LaneId = RuntimeLane.LaneHandle.LaneId;
		OutSample.EdgeId = RuntimeLane.LaneHandle.EdgeId;
		OutSample.LaneIndex = RuntimeLane.LaneHandle.LaneIndex;
		OutSample.Distance2DCm = CandidateDistance2DCm;
		OutSample.DistanceAlongLaneM = DistanceAlongSplineCm * 0.01f;
		OutSample.WorldTransform = CandidateTransform;
	}

	return bFoundSample;
}

bool ASumoRoadNetworkActor::ExportLaneSamplesToCsv(const FString& OutputCsvPath, float StepMeters) const
{
	if (OutputCsvPath.IsEmpty())
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportLaneSamplesToCsv failed: OutputCsvPath is empty."));
		return false;
	}

	const FString ResolvedPath = ResolveOutputPath(OutputCsvPath);
	if (!EnsureOutputDirectory(ResolvedPath))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportLaneSamplesToCsv failed: cannot create output directory for path=%s"), *ResolvedPath);
		return false;
	}

	FString CsvContent;
	int32 ExportedLaneCount = 0;
	int32 ExportedPointCount = 0;
	if (!BuildLaneCenterSamplesCsv(StepMeters, CsvContent, ExportedLaneCount, ExportedPointCount))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportLaneSamplesToCsv failed: no valid lane samples (runtime lanes may be empty)."));
		return false;
	}

	if (!FFileHelper::SaveStringToFile(CsvContent, *ResolvedPath))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportLaneSamplesToCsv failed: cannot write file path=%s"), *ResolvedPath);
		return false;
	}

	UE_LOG(
		LogSumoImporter,
		Log,
		TEXT("Exported SUMO lane samples: lanes=%d points=%d path=%s"),
		ExportedLaneCount,
		ExportedPointCount,
		*ResolvedPath);
	return true;
}

bool ASumoRoadNetworkActor::ExportTrafficBundleToCsv(const FString& OutputDir, float SampleStepMeters) const
{
	if (OutputDir.IsEmpty())
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: OutputDir is empty."));
		return false;
	}

	if (RuntimeLanes.IsEmpty())
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: RuntimeLanes is empty. Import/build SUMO network first."));
		return false;
	}

	FString ResolvedOutputDir = ResolveOutputPath(OutputDir);
	if (!IFileManager::Get().MakeDirectory(*ResolvedOutputDir, true))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: cannot create output directory=%s"), *ResolvedOutputDir);
		return false;
	}

	const FString CenterSamplesPath = FPaths::Combine(ResolvedOutputDir, TEXT("lane_center_samples.csv"));
	const FString ConnectionsPath = FPaths::Combine(ResolvedOutputDir, TEXT("lane_connections.csv"));
	const FString MetaPath = FPaths::Combine(ResolvedOutputDir, TEXT("lane_meta.csv"));

	FString CenterCsv;
	int32 ExportedLaneCount = 0;
	int32 ExportedPointCount = 0;
	if (!BuildLaneCenterSamplesCsv(SampleStepMeters, CenterCsv, ExportedLaneCount, ExportedPointCount))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: no valid lane center samples."));
		return false;
	}
	if (!FFileHelper::SaveStringToFile(CenterCsv, *CenterSamplesPath))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: cannot write %s"), *CenterSamplesPath);
		return false;
	}

	FString ConnectionsCsv = TEXT("from_edge,from_lane,to_edge,to_lane,via_lane\n");
	for (const FSumoConnectionData& Connection : RawConnections)
	{
		ConnectionsCsv += FString::Printf(
			TEXT("%s,%d,%s,%d,%s\n"),
			*EscapeCsvField(Connection.FromEdge),
			Connection.FromLane,
			*EscapeCsvField(Connection.ToEdge),
			Connection.ToLane,
			*EscapeCsvField(Connection.ViaLane));
	}
	if (!FFileHelper::SaveStringToFile(ConnectionsCsv, *ConnectionsPath))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: cannot write %s"), *ConnectionsPath);
		return false;
	}

	FString MetaCsv = TEXT("edge_id,lane_id,lane_index,length_m,speed_mps,edge_type,road_weight\n");
	for (const FSumoLaneRuntimeData& RuntimeLane : RuntimeLanes)
	{
		MetaCsv += FString::Printf(
			TEXT("%s,%s,%d,%.3f,%.3f,%s,%.3f\n"),
			*EscapeCsvField(RuntimeLane.LaneHandle.EdgeId),
			*EscapeCsvField(RuntimeLane.LaneHandle.LaneId),
			RuntimeLane.LaneHandle.LaneIndex,
			RuntimeLane.LaneHandle.LengthM,
			RuntimeLane.LaneHandle.SpeedMps,
			*EscapeCsvField(RuntimeLane.LaneHandle.EdgeType),
			SanitizeRoadWeight(RuntimeLane.LaneHandle.RoadWeight));
	}
	if (!FFileHelper::SaveStringToFile(MetaCsv, *MetaPath))
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv failed: cannot write %s"), *MetaPath);
		return false;
	}

	if (RawConnections.IsEmpty())
	{
		UE_LOG(LogSumoImporter, Warning, TEXT("ExportTrafficBundleToCsv: no lane connections available, lane_connections.csv contains only header."));
	}

	UE_LOG(
		LogSumoImporter,
		Log,
		TEXT("Exported SUMO traffic bundle: lanes=%d points=%d connections=%d dir=%s"),
		ExportedLaneCount,
		ExportedPointCount,
		RawConnections.Num(),
		*ResolvedOutputDir);
	return true;
}

void ASumoRoadNetworkActor::ExportTrafficBundleNow()
{
	if (!ExportTrafficBundleToCsv(TrafficBundleOutputDir, TrafficBundleSampleStepMeters))
	{
		UE_LOG(
			LogSumoImporter,
			Warning,
			TEXT("ExportTrafficBundleNow failed. output_dir=%s sample_step=%.3f"),
			*TrafficBundleOutputDir,
			TrafficBundleSampleStepMeters);
	}
}

bool ASumoRoadNetworkActor::BuildLaneCenterSamplesCsv(
	float StepMeters,
	FString& OutCsvContent,
	int32& OutExportedLaneCount,
	int32& OutExportedPointCount) const
{
	OutCsvContent = TEXT("edge_id,lane_id,lane_index,s_m,x_m,y_m,z_m,yaw_deg\n");
	OutExportedLaneCount = 0;
	OutExportedPointCount = 0;

	if (RuntimeLanes.IsEmpty())
	{
		return false;
	}

	const float SafeStepMeters = FMath::Max(0.1f, StepMeters);
	for (const FSumoLaneRuntimeData& RuntimeLane : RuntimeLanes)
	{
		if (!IsValid(RuntimeLane.SplineComponent))
		{
			continue;
		}

		const float SplineLengthCm = RuntimeLane.SplineComponent->GetSplineLength();
		if (SplineLengthCm <= KINDA_SMALL_NUMBER)
		{
			continue;
		}

		const float LaneLengthM = RuntimeLane.LaneHandle.LengthM > 0.0f
			? RuntimeLane.LaneHandle.LengthM
			: (SplineLengthCm * 0.01f);
		const int32 SegmentCount = FMath::Max(1, FMath::CeilToInt(LaneLengthM / SafeStepMeters));
		const FString EscapedEdgeId = EscapeCsvField(RuntimeLane.LaneHandle.EdgeId);
		const FString EscapedLaneId = EscapeCsvField(RuntimeLane.LaneHandle.LaneId);

		for (int32 SegmentIndex = 0; SegmentIndex <= SegmentCount; ++SegmentIndex)
		{
			const float SInMeters = (SegmentIndex == SegmentCount)
				? LaneLengthM
				: (SegmentIndex * SafeStepMeters);
			const float DistanceCm = FMath::Clamp(SInMeters * 100.0f, 0.0f, SplineLengthCm);
			const FVector LocationCm = RuntimeLane.SplineComponent->GetLocationAtDistanceAlongSpline(DistanceCm, ESplineCoordinateSpace::World);
			const FVector Direction = RuntimeLane.SplineComponent->GetDirectionAtDistanceAlongSpline(DistanceCm, ESplineCoordinateSpace::World);
			const FRotator Rotation = Direction.Rotation();

			OutCsvContent += FString::Printf(
				TEXT("%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n"),
				*EscapedEdgeId,
				*EscapedLaneId,
				RuntimeLane.LaneHandle.LaneIndex,
				SInMeters,
				LocationCm.X * 0.01f,
				LocationCm.Y * 0.01f,
				LocationCm.Z * 0.01f,
				Rotation.Yaw);
			++OutExportedPointCount;
		}

		++OutExportedLaneCount;
	}

	return OutExportedPointCount > 0;
}

FString ASumoRoadNetworkActor::MakeEdgeLaneKey(const FString& EdgeId, int32 LaneIndex)
{
	return FString::Printf(TEXT("%s::%d"), *EdgeId, LaneIndex);
}

bool ASumoRoadNetworkActor::ResolveLaneRuntimeIndex(const FString& EdgeId, int32 LaneIndex, int32& OutRuntimeIndex) const
{
	const int32* FoundIndex = EdgeLaneToRuntimeIndex.Find(MakeEdgeLaneKey(EdgeId, LaneIndex));
	if (FoundIndex == nullptr || !RuntimeLanes.IsValidIndex(*FoundIndex))
	{
		return false;
	}

	OutRuntimeIndex = *FoundIndex;
	return true;
}
