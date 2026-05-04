#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "AeroBridgeRuntimeSettings.generated.h"

UCLASS(Config = Game, DefaultConfig, meta = (DisplayName = "Aero Bridge Runtime"))
class AEROBRIDGERUNTIME_API UAeroBridgeRuntimeSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UAeroBridgeRuntimeSettings();

	virtual FName GetCategoryName() const override;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Paths")
	FString LowAltitudeConfigRoot;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Paths")
	FString AssetCatalogRelativePath;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Paths")
	FString WeatherProfilesRelativePath;

	UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Paths")
	FString MapsRelativeRoot;
};
