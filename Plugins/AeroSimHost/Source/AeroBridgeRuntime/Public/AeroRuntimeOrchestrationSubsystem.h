#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroRuntimeOrchestrationSubsystem.generated.h"

class AActor;
class ASimModeBase;
class UAnimationAsset;
class UPedestrianWorldSubsystem;

UCLASS()
class AEROBRIDGERUNTIME_API UAeroRuntimeOrchestrationSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Deinitialize() override;

	bool SpawnPedestrian(const FString& PedId, const FVector& WorldLocationCm, float YawDeg, FName VariantId, FString& OutError, bool bUseProvidedGroundPoint = false) const;
	bool ResetPedestrian(const FString& PedId, const FVector& WorldLocationCm, float YawDeg, FString& OutError, bool bUseProvidedGroundPoint = false) const;
	bool SetPedestrianFramePose(const FString& PedId, const FVector& WorldLocationCm, float YawDeg, bool bWalking, float SpeedCmPerSec, FString& OutError, bool bUseProvidedGroundPoint = false) const;
	bool SetPedestrianTarget(const FString& PedId, const FVector& TargetWorldCm, float SpeedCmPerSec, FString& OutError) const;
	bool ObservePedestrian(const FString& PedId, FName StartSection, FString& OutError) const;
	bool PlayPedestrianAnimation(const FString& PedId, const FString& AnimationAssetPath, FName StartSection, float PlayRate, int32 LoopCount, FString& OutError) const;
	bool CommitPedestrianCross(const FString& PedId, const FVector& TargetWorldCm, float SpeedCmPerSec, FString& OutError) const;
	bool StopPedestrian(const FString& PedId, FString& OutError) const;
	bool SetPedestrianVariant(const FString& PedId, FName VariantId, FString& OutError) const;
	bool ReleasePedestrian(const FString& PedId, FString& OutError) const;
	bool SpawnCrowd(const FCrowdSpawnRequest& Request, FCrowdSpawnResult& OutResult, FString& OutError) const;
	bool ClearCrowdGroup(FName GroupId, FString& OutError) const;
	bool RespawnCrowd(FName GroupId, int32 NewSeed, FCrowdSpawnResult& OutResult, FString& OutError) const;
	AActor* ResolvePedestrianActor(const FString& PedId) const;

private:
	UPedestrianWorldSubsystem* ResolvePedestrianSubsystem(FString& OutError) const;
	ASimModeBase* ResolveSimModeActor() const;
	bool RegisterActorWithAirSimInstanceSegmentation(AActor* Actor, const FString& Context, int32 ObjectId) const;
};
