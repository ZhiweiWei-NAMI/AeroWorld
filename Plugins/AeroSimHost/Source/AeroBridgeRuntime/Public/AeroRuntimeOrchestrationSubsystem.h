#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroRuntimeOrchestrationSubsystem.generated.h"

class AActor;
class APawn;
class ASimModeBase;
class UAnimationAsset;
class UPedestrianWorldSubsystem;

enum class EAeroRuntimeMoveState : uint8
{
	Idle,
	Running,
	Succeeded,
	Failed,
	Cancelled
};

struct FAeroRuntimeAirSimCapabilities
{
	FString SimModeName = TEXT("Unavailable");
	bool bSupportsMultirotors = false;
};

struct FAeroRuntimeMoveStatus
{
	EAeroRuntimeMoveState State = EAeroRuntimeMoveState::Idle;
	FString Message;
	FVector TargetWorldCm = FVector::ZeroVector;
	float VelocityMps = 0.0f;
};

UCLASS()
class AEROBRIDGERUNTIME_API UAeroRuntimeOrchestrationSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Deinitialize() override;

	bool GetCurrentAirSimCapabilities(FAeroRuntimeAirSimCapabilities& OutCapabilities) const;

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

	bool CreateRuntimeMultirotor(
		const FString& VehicleName,
		const FVector& WorldLocationCm,
		const FRotator& WorldRotation,
		FString& OutError);
	bool EnableApiControl(const FString& VehicleName, bool bEnable, FString& OutError) const;
	bool ArmIfSupported(const FString& VehicleName, FString& OutError) const;
	bool MoveMultirotorToPosition(const FString& VehicleName, const FVector& TargetWorldCm, float VelocityMps, FString& OutError);
	bool GetMultirotorMoveStatus(const FString& VehicleName, FAeroRuntimeMoveStatus& OutStatus, FString& OutError) const;
	bool RemoveRuntimeVehicle(const FString& VehicleName, FString& OutError);
	APawn* ResolveSpawnedPawnByVehicleName(const FString& VehicleName) const;
	bool GetVehiclePose(const FString& VehicleName, FVector& OutWorldLocationCm, FRotator& OutWorldRotation, FString& OutError) const;

private:
	struct FMultirotorMoveJobState;

	UPedestrianWorldSubsystem* ResolvePedestrianSubsystem(FString& OutError) const;
	ASimModeBase* ResolveSimModeActor() const;
	bool CancelTrackedMove(const FString& VehicleName, bool bWaitForCompletion, FString& OutError);

private:
	mutable FCriticalSection MultirotorMoveJobsMutex;
	TMap<FString, TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe>> MultirotorMoveJobs;
	TMap<FString, FVector> RuntimeMultirotorSpawnWorldCm;
	uint64 NextMultirotorMoveCommandId = 1;
};
