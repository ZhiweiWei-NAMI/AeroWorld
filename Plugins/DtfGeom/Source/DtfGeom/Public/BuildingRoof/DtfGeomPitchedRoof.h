// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "DtfGeomStructsDefine.h"
#include "GeometryScript/GeometryScriptTypes.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "DtfGeomPitchedRoof.generated.h"

class UDynamicMesh;

UCLASS()
class UDtfGeomPitchedRoofGenerator : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:

	/**
	 * 需要保障 InPolyline 点逆时针缠绕，且首尾点重合
	 * 使用完 OutTargetMesh 后，需及时使用 DestroyUObject 释放。注意这里不是使用 ReleaseComputeMesh 或 FreeComputeMesh
	 */
	UFUNCTION(BlueprintCallable, Category="DtfGeom")
	static bool GetPicthedPartition(
		const TArray<FVector>& InPolyline,
		const float InAngle,
		FGeometryScriptSimpleMeshBuffers& OutBuffers,
		TArray<FDtfGeomPolyline>& OutParts,
		TArray<FDtfGeomPolyline>& OutIntersectionPolylinesBetweenParts
		);

	/**
	 * 获取角度符合的边的数组
	 * @param MinAngle 一条 Edge 所关联的两个三角面的法线间夹角，大于该角度的，该 EdgeID 将会被填充到 OutEdgeIndices 中
	 * 
	 */
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetAngleValidEdgeIndices(UDynamicMesh* TargetMesh,const float MinAngle, FGeometryScriptIndexList& OutEdgeIndices);

	/** 求取不用的 Polylines 之间的重叠边 */
	UFUNCTION(BlueprintCallable, Category="DtfGeom")
	static TArray<FDtfGeomPolyline> GetPartsIntersectionPolylines(const TArray<FDtfGeomPolyline>& InParts);

};