// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "ModelingObjectsCreationAPI.h"
#include "DynamicMesh/DynamicMeshAttributeSet.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TargetInterfaces/MaterialProvider.h"
#include "DtfGeomMeshHelper.generated.h"

class UDynamicMeshComponent;
class AStaticMeshActor;

UCLASS()
class UDtfGeomMeshHelper : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
		
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "DtfGeom", meta = (WorldContext = "WorldContextObject"))
	static void ConvertMesh(UObject* WorldContextObject, ADynamicMeshActor* InActor, AStaticMeshActor*& OutMergedActor);
};