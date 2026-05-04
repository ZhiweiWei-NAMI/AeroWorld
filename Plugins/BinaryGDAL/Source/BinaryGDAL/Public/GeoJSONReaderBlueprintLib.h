// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"

#include "GeoJSONReaderBlueprintLib.generated.h"

UCLASS()
class BINARYGDAL_API UGeoJSONReaderBlueprintLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintCallable, Category = "Default")
	static void ReadGeoJSON(const FString& FilePath);
};