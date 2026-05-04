#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SumoTypes.h"
#include "SumoRoadNetworkActor.generated.h"

class USplineComponent;
class USceneComponent;

USTRUCT()
struct FSumoLaneRuntimeData
{
	GENERATED_BODY()

	UPROPERTY()
	FSumoLaneHandle LaneHandle;

	UPROPERTY()
	TObjectPtr<USplineComponent> SplineComponent = nullptr;
};

UCLASS(BlueprintType)
class SUMOIMPORTER_API ASumoRoadNetworkActor : public AActor
{
	GENERATED_BODY()

public:
	ASumoRoadNetworkActor();

	void ResetNetwork();
	bool AddLane(const FSumoLaneData& LaneData, const FString& EdgeType, float RoadWeight, const TArray<FVector>& LanePointsWorld);
	void AddJunctionDebugPolygon(const FSumoJunctionData& JunctionData, const TArray<FVector>& PolygonPointsWorld);
	void SetConnections(const TArray<FSumoConnectionData>& InConnections);

	UFUNCTION(BlueprintCallable, Category = "SUMO|Query")
	bool FindLaneById(const FString& LaneId, FSumoLaneHandle& OutLane) const;

	UFUNCTION(BlueprintCallable, Category = "SUMO|Query")
	void FindLanesByEdge(const FString& EdgeId, TArray<FSumoLaneHandle>& OutLanes) const;

	UFUNCTION(BlueprintCallable, Category = "SUMO|Query")
	bool SampleTransformByEdgeLane(const FString& EdgeId, int32 LaneIndex, float SInMeters, FTransform& OutTransform, bool& bWasClamped) const;

	UFUNCTION(BlueprintCallable, Category = "SUMO|Query")
	void GetSuccessors(const FString& FromEdge, int32 FromLane, TArray<FSumoConnectionData>& OutSuccessors) const;

	UFUNCTION(BlueprintCallable, Category = "SUMO|Query")
	bool FindNearestLaneSample(const FVector& QueryWorldCm, FSumoNearestLaneSample& OutSample) const;

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "SUMO|Export")
	bool ExportLaneSamplesToCsv(const FString& OutputCsvPath, float StepMeters = 1.0f) const;

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "SUMO|Export")
	bool ExportTrafficBundleToCsv(const FString& OutputDir, float SampleStepMeters = 1.0f) const;

	UFUNCTION(CallInEditor, Category = "SUMO|Export")
	void ExportTrafficBundleNow();

	UFUNCTION(BlueprintPure, Category = "SUMO|Query")
	int32 GetLaneCount() const
	{
		return RuntimeLanes.Num();
	}

private:
	static FString MakeEdgeLaneKey(const FString& EdgeId, int32 LaneIndex);
	bool ResolveLaneRuntimeIndex(const FString& EdgeId, int32 LaneIndex, int32& OutRuntimeIndex) const;
	bool BuildLaneCenterSamplesCsv(float StepMeters, FString& OutCsvContent, int32& OutExportedLaneCount, int32& OutExportedPointCount) const;

	UPROPERTY(VisibleAnywhere, Category = "SUMO")
	TObjectPtr<USceneComponent> SceneRoot;

	UPROPERTY(VisibleAnywhere, Category = "SUMO")
	TArray<TObjectPtr<USplineComponent>> LaneSplineComponents;

	UPROPERTY(VisibleAnywhere, Category = "SUMO")
	TArray<TObjectPtr<USplineComponent>> JunctionDebugSplines;

	UPROPERTY(EditAnywhere, Category = "SUMO|Export", meta = (ToolTip = "Relative path is resolved under Project/Saved."))
	FString TrafficBundleOutputDir = TEXT("SUMO/traffic_bundle");

	UPROPERTY(EditAnywhere, Category = "SUMO|Export", meta = (ClampMin = "0.1", UIMin = "0.1"))
	float TrafficBundleSampleStepMeters = 1.0f;

	UPROPERTY()
	TArray<FSumoLaneRuntimeData> RuntimeLanes;

	UPROPERTY()
	TArray<FSumoConnectionData> RawConnections;

	TMap<FString, int32> LaneIdToRuntimeIndex;
	TMap<FString, TArray<int32>> EdgeIdToRuntimeIndices;
	TMap<FString, int32> EdgeLaneToRuntimeIndex;
	TMap<FString, TArray<FSumoConnectionData>> SuccessorConnectionsByLaneKey;
};
