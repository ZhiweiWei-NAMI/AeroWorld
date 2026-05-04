#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "GameFramework/Actor.h"
#include "CrowdSpawnerActor.generated.h"

class UBoxComponent;
class UCrowdAppearancePool;
class UCrowdRoleProfile;

UCLASS(BlueprintType, Blueprintable)
class PEDESTRIANRUNTIME_API ACrowdSpawnerActor : public AActor
{
	GENERATED_BODY()

public:
	ACrowdSpawnerActor();
	virtual void BeginPlay() override;

	UFUNCTION(BlueprintCallable, Category = "Crowd")
	FCrowdSpawnResult SpawnCrowdNow();

	UFUNCTION(BlueprintCallable, Category = "Crowd")
	bool ClearSpawnedCrowd();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Crowd")
	TObjectPtr<UBoxComponent> SpawnZone = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	int32 Count = 30;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	int32 Seed = 1001;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	FName GroupId = FName(TEXT("crowd.cityops.default"));

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	bool bAutoSpawnOnBeginPlay = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	ECrowdYawPolicy YawPolicy = ECrowdYawPolicy::Random;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	float FixedYawDeg = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	TObjectPtr<UCrowdAppearancePool> AppearancePool = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd")
	TObjectPtr<UCrowdRoleProfile> RoleProfile = nullptr;

private:
	FCrowdSpawnRequest BuildRequest() const;
};
