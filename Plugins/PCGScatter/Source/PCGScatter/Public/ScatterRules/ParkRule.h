
// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include <optional>

#include "CoreMinimal.h"
#include "DynamicMeshActor.h"
#include "GeometryScript/GeometryScriptTypes.h"
#include "PCGScatterDataCollection.h"
#include "ParkRule.generated.h"

struct FDtfGeomPolyline;
class USplineComponent;
class UInstancedStaticMeshComponent;

UCLASS()
class PCGSCATTER_API AParkScatterRule : public ADynamicMeshActor
{
	GENERATED_BODY()

public:
	AParkScatterRule();

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	TArray<FVector> GetSplineOriginPoints(USplineComponent* InSpline);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	bool GetDividedPointsOnSpline(USplineComponent* InSpline, const float MaxSquareDistanceFromSpline, TArray<FVector>& OutPoints);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void GetSplineInteriorPoints(
		USplineComponent* InSpline,
		const float InSamplingRadius,
		const int InMaxNumSamples,
		const int InRandomSeed,
		const float InSubSampleDensity,
		TArray<FVector>& OutPoints);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void DrawDebugPoints(const TArray<FVector>& InPoints);

	/** 返回在 Spline 上离 InPoint 最近点的 InputKey, 如果 Spline 是 ClosedLoop, 且找出的最近点的位置在曲线最后一个点和曲线第一个点之间，返回值将是 [LastPointInputKey, LastPointInputKey+1) 区间内的值*/
	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void CalcMinDistToSpline(USplineComponent* InSpline, const FVector& InPoint, float& OutDistance, float& OutSplineInputKey);

	/**
	 * 生成 Park 的底面, 用于 PCG 点位投影
	 * @param ProjectionTargetTag 
	 */
	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	UDynamicMesh* GenerateBottomPolygonByProjectionTargetTag(USplineComponent* InSpline, const FString& ProjectionTargetTag);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	UDynamicMesh* GenerateBottomPolygonByProjectionTargetMesh(USplineComponent* InSpline, UDynamicMesh* ProjectionTargetMesh);

	static std::optional<int> FindIntervalForNumber(float Number, const TArray<TPair<float, float>>& Intervals);
	static TArray<TPair<float, float>> SplitTValusToOrderedPairs(const TArray<float>& InTValues);

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ParkScatterRule")
	void DesignPark(
		const TArray<FVector>& InContourPoints,
		const int32 InDesiredInteriorSamplePoints,
		UDynamicMesh* InProjectionTerrain,
		TArray<FDtfGeomPolyline>& OutRoads,
		const FString& ProjectionTargetTag = TEXT(""));

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void GetDividedPointsOnSplineAndInterior(const TArray<FVector>& InContourPoints,
		TArray<FVector>& OutPointsOnSpline,
		TArray<FVector>& OutPointsInterior,
		const float InMaxSquareDistanceAlongContour = 100.f,
		const float InSmaplingRadius = 100.f,
		const int32 InMaxNumInteriorSamples = 100,
		const int InRandomSeed = 0,
		const float InSubSampleDensity = 1.f
		);

	/**
	 * 
	 * @param VoronoiSites 所有用于构建 Voronoi 形状的点集
	 * @param VoronoiOptions 
	 * @return 返回生成的每一个 CellMeshes
	 */
	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	TArray<UDynamicMesh*>& GenerateVoronoiShape(const TArray<FVector>& VoronoiSites,
		FGeometryScriptVoronoiOptions VoronoiOptions
		);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	TArray<UDynamicMesh*>& GroupedCellsByNormalizedTValues(TArray<FDtfGeomPolyline>& OutGroupBoundaries);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void FindPolylinesAmongGroups(
		const TArray<FDtfGeomPolyline>& InBoundaryPolylines,
		TArray<FDtfGeomPolyline>& OutIntersectionPolylines,
		TMap<FVector, FGeometryScriptIndexList>& OutCrossPointToIntersectedPolylineIndices
		);

	void FindNearestRayIntersectionWithTerrain(TArray<FVector>& OutPoints, bool bResetTerrainDMPointer = false);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void GenParkRoad(const TArray<FDtfGeomPolyline>& InPolylines);

	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	void ClearCache();

	UFUNCTION(CallInEditor, Category = "ParkScatterRule")
	void AfterParkActorGenerated(FPCGDataCollection& InputCollection);
	
	UFUNCTION(BlueprintCallable, Category = "ParkScatterRule")
	UDynamicMesh* GetBottomDynamicMesh();

public:

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UPCGComponent* ParkPCGScatterRule = nullptr;
	
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UInstancedStaticMeshComponent* DebugInstancedMeshComp = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UDynamicMeshComponent* DebugVoronoiDMC = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UDynamicMeshComponent* DebugTerrainDMC = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UDynamicMeshComponent* DebugBottomDMC = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UDynamicMeshComponent* DebugWaterDMC = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	UDynamicMeshComponent* DebugParkRoadDMC = nullptr;

	UPROPERTY(BlueprintReadWrite, EditAnywhere /*, meta = (ExposeOnSpawn = true)*/, Category = "ParkScatterRule")
	UDynamicMesh* TerrainDM = nullptr;

	/** 原始的绿地数据外轮廓 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	USplineComponent* DebugOuterContourSpline = nullptr;

	/** 为方便处理，填充归一化的分割值，例如 [0.1, 0.3, 0,5, 0.999...] */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<float> NormalizedTValues = { 0.3, 0.5, 0.68, 0.8 };

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<UMaterialInterface*> PresetMats;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<FVector> AllPoints;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<FVector> PointsOnSpline;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<FVector> PointsInteriorSpline;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<USplineComponent*> DebugInteriorBoundarySplines;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<UDynamicMesh*> CachedCellMeshes;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<UDynamicMesh*> CachedGroupMeshes;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<USplineComponent*> DebugIntersectionSplines;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<class UTextRenderComponent*> DebugTextRenderComps;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	FString TerrainActorTag = TEXT("terrain");

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	FString GeneratedPCGStampActorTag = TEXT("PCGPlant");

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "ParkScatterRule")
	TArray<FName> PostGenerateFunctionNames;

private:
	
	static void SplitTransformArrayEelements(const TArray<FTransform>& InElements, TArray<FVector>& OutPositions, TArray<FRotator>& OutRotations, TArray<FVector>& OutScales);

};