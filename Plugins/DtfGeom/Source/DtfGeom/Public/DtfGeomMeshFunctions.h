// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "DtfGeomMeshFunctions.generated.h"

UCLASS()
class DTFGEOM_API UDtfGeomMeshFunctions : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	AppendDtfGeomVoronoiDiagram2D(
		UDynamicMesh* TargetMesh,
		FGeometryScriptPrimitiveOptions PrimitiveOptions,
		FTransform Transform,
		const TArray<FVector2D>& VoronoiSites,
		FGeometryScriptVoronoiOptions VoronoiOptions,
		bool bOutputPolyLine,
		TArray<int32>& CellMembers,
		TArray<FDtfGeomPolyline>& OutPolylines,
		UGeometryScriptDebug* Debug = nullptr);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static UPARAM(DisplayName = "Target Mesh") UDynamicMesh*
	GetDtfGeomPolygroupIDsInMesh(
		UDynamicMesh* TargetMesh,
		FGeometryScriptGroupLayer GroupLayer,
		UPARAM(ref, DisplayName="PolyGroup IDs Out") FGeometryScriptIndexList& PolygroupIDsOut,
		TArray<int>& IndexList);
};