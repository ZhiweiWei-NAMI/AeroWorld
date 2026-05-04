// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"
#include "DtfGeomMeshConvert.generated.h"

UCLASS()
class DTFGEOM_API UDtfGeomMeshConvert : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintCallable, Category = "DtfGeom", meta = (WorldContext = "WorldContextObject"))
	static void ConvertToInstancedMeshActor(UObject* WorldContextObject, AActor* InSourceActor);
	
};
