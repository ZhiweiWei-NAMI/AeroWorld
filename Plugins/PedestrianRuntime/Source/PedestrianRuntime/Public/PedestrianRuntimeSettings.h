#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "PedestrianRuntimeSettings.generated.h"

class APedestrianCharacter;
class UCrowdAppearancePool;
class UCrowdRoleProfile;

UCLASS(Config = Game, DefaultConfig, meta = (DisplayName = "Pedestrian Runtime"))
class PEDESTRIANRUNTIME_API UPedestrianRuntimeSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UPedestrianRuntimeSettings();

	virtual FName GetCategoryName() const override;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Spawn")
	TSoftClassPtr<APedestrianCharacter> DefaultPedestrianClass;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Spawn")
	FName DefaultSpawnVariantId = NAME_None;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	TSoftObjectPtr<UCrowdAppearancePool> DefaultCrowdAppearancePool;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	TSoftObjectPtr<UCrowdRoleProfile> DefaultCrowdRoleProfile;
};
