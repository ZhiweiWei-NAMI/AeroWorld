#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "Engine/DataAsset.h"
#include "CrowdAppearancePool.generated.h"

UCLASS(BlueprintType)
class PEDESTRIANRUNTIME_API UCrowdAppearancePool : public UDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	TArray<FCrowdAppearanceEntry> Entries;
};
