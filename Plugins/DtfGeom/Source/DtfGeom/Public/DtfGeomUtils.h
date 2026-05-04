// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "UDynamicMesh.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "DtfGeomUtils.generated.h"

struct FGeometryScriptIndexList;
struct FDtfGeomPolyline;

UCLASS()
class DTFGEOM_API UDtfGeomUtils : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static FBox GetBoundingBox(const TArray<FVector>& Vertices);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static TArray<int> GetOpenBorderEdgesID(UDynamicMesh* TargetMesh);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static FVector2f GetEdgeVerticesID(UDynamicMesh* TargetMesh, int EdgeID);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static FVector2f GetEdgeTrianglesID(UDynamicMesh* TargetMesh, int EdgeID);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static int GetTriangleGroupID(UDynamicMesh* TargetMesh, int tID);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetGroupBoundaryVertices(
		UDynamicMesh* TargetMesh,
		TArray<int32>& GroupBoundaryVerticesID);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetGroupBoundaryEdges(
		UDynamicMesh* TargetMesh,
		TArray<int32>& GroupBoundaryEdgesID);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void JoinPolyline(const TArray<FDtfGeomPolyline>& InPolylines, TArray<FDtfGeomPolyline>& OutPolylines);

	/**
	 * 
	 * @param bRandomChooseP true: 随机选一个点；false: 所有点平均XY坐标
	 * @param InVertices 
	 * @param Tolerance 
	 * @param OutVertices 
	 * @param OutIndices 
	 * @param Volance 
	 */
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void CullDuplicatePoints(bool bRandomChooseP, const TArray<FVector>& InVertices, double Tolerance, TArray<FVector>& OutVertices, TArray<int>& OutIndices, int32 Volance);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void SplitPolyline(const FDtfGeomPolyline& InPolyline, const TArray<float>& InTValues, TArray<FDtfGeomPolyline>& OutPolylines);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static double CalcMinDistanceToPolyline(const FVector& P, const TArray<FDtfGeomPolyline>& InPolylines);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static float CalcAngleBetweenVectors(const FVector& UnitVec1, const FVector& UnitVec2);

	/**
	 * 计算多边形每个点位处的内角。若点位序列中首尾点是重合的，在计算内角序列时会将首尾点看做一个去计算内角，计算得到的值会在首尾点处分别记录下来。
	 * 注意，所有点需要在同一个平面上。
	 * @param InPolyline 点位序列
	 * @param Values 内角序列
	 */
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void CalcPolygonInteriorCorners(const FDtfGeomPolyline& InPolyline, TArray<float>& Values);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void ComputeConvexHull2D(const TArray<FVector>& InPoints, TArray<FVector>& OutPoints);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetBoundaryEdgeVertices(
		UDynamicMesh* TargetMesh,
		TArray<FVector>& OutPoints);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetBoundaryEdgeVerticesInOrder(
		UDynamicMesh* TargetMesh,
		TArray<FVector>& OutPoints);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static float GetPolygonArea(const TArray<FVector>& InPoints);
	
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static FVector GetPolygonCentroid(const TArray<FVector>& InPoints);

	/**
	 * 通过将光线投射 MaxDistance 计算交点，来测试点是否位于给定多边形内
	 * @param PolygonPoints 
	 * @param Point 
	 * @param MaxDistance 
	 * @return 
	 */
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static bool ISPointInsidePolygon2D(const TArray<FVector2D>& PolygonPoints, const FVector2D& Point, double MaxDistance);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static TArray<FVector2D> VectorArrayToVector2DArray(const TArray<FVector>& InPoints);

	/** 获取一个 DynamicMesh 中两个边界重叠的 Polygroup 间，重叠的点，按点的顺序返回。调用者需要确保 TargetMesh 在 Polygroup 边界处是焊接过的 */
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static TArray<FVector> GetPolylineBetweenTwoIntersectionPolygroups(UDynamicMesh* TargetMesh);
};