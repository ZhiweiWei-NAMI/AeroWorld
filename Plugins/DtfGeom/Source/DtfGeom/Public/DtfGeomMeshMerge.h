// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "ModelingObjectsCreationAPI.h"
#include "DynamicMesh/DynamicMeshAttributeSet.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TargetInterfaces/MaterialProvider.h"
#include "DtfGeomMeshMerge.generated.h"

class UDynamicMeshComponent;
class AStaticMeshActor;

UCLASS()
class DTFGEOM_API UDtfGeomMeshMerge : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:

	 static void DtfGeom_GetDmcMaterialSet(UDynamicMeshComponent* DMC, FComponentMaterialSet& MaterialSetOut);
	/**
	 * 
	 * @param InDMCs 所有待合并的 DynamicMeshComponent
	 * @param NewMaterialOut 所有用到的材质的数组
	 * @param MaterialIDRemapsOut 每个 DynamicMeshComponent 用到的材质的记录，其中存储的 Indices 对应于 NewMaterialOut 中的元素
	 */
	static void DtfGeom_BuildCombinedMaterialSet(
		TArray<UDynamicMeshComponent*>& InDMCs,
		TArray<UMaterialInterface*>& NewMaterialOut,
		TArray<TArray<int32>>& MaterialIDRemapsOut);

	static void DtfGeom_SetNewMaterialID(
		TArray<UDynamicMeshComponent*>& InDMCs,
		int32 ComponentIdx,
		UE::Geometry::FDynamicMeshMaterialAttribute* MatAttrib,
		int32 TID,
		TArray<TArray<int32>>& MaterialIDRemaps,
		TArray<UMaterialInterface*>& AllMaterials);

	static FCreateMeshObjectResult DtfGeom_CreateDynamicMeshActor(FCreateMeshObjectParams&& CreateMeshParams);

	static TArray<UMaterialInterface*> DtfGeom_FilterMaterials(const TArray<UMaterialInterface*>& MaterialsIn);
	
	UFUNCTION(BlueprintCallable,CallInEditor, Category = "DtfGeom", meta=(WorldContext="WorldContextObject"))
	static void MergeMesh(UObject* WorldContextObject, UPARAM(ref)TArray<AActor*>& InActors, AActor*& OutMergedActor);
};