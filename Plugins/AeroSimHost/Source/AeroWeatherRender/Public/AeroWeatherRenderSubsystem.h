#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroWeatherRenderSubsystem.generated.h"

class FJsonObject;

UCLASS()
class AEROWEATHERRENDER_API UAeroWeatherRenderSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;

	bool LoadProfiles(const FString& ProfilesPath, FString& OutError);
	TSharedPtr<FJsonObject> ApplyWeather(const TSharedPtr<FJsonObject>& Payload, FString& OutError);

private:
	TSharedPtr<FJsonObject> ProfilesDocument;
};
