#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "Engine/DataAsset.h"
#include "CrowdRoleProfile.generated.h"

UCLASS(BlueprintType)
class PEDESTRIANRUNTIME_API UCrowdRoleProfile : public UDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	TArray<ECrowdGender> AllowedGenders;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	TArray<ECrowdAgeGroup> AllowedAgeGroups;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	TArray<FName> RequiredTags;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	TArray<FName> BlockedTags;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	TArray<FCrowdTagWeightMultiplier> WeightMultipliers;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	int32 CountOverride = -1;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	FName DefaultBehaviorMode = FName(TEXT("idle"));

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role", meta = (ClampMin = "0.0"))
	float DefaultSpawnRadius = 1200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role", meta = (ClampMin = "0.0"))
	float DefaultMinSpacing = 120.0f;
};
