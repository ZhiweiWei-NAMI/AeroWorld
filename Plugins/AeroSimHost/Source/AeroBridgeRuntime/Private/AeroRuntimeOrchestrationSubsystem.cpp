#include "AeroRuntimeOrchestrationSubsystem.h"

#include "Animation/AnimationAsset.h"
#include "PedestrianCharacter.h"
#include "PedestrianWorldSubsystem.h"
#include "UObject/SoftObjectPath.h"

bool UAeroRuntimeOrchestrationSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroRuntimeOrchestrationSubsystem::Deinitialize()
{
	Super::Deinitialize();
}

bool UAeroRuntimeOrchestrationSubsystem::SpawnPedestrian(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	const FName VariantId,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSpawn(PedId, WorldLocationCm, YawDeg, VariantId, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to spawn pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ResetPedestrian(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecReset(PedId, WorldLocationCm, YawDeg, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to reset pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianFramePose(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	const bool bWalking,
	const float SpeedCmPerSec,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetFramePose(PedId, WorldLocationCm, YawDeg, bWalking, SpeedCmPerSec, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to set frame pose for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianTarget(
	const FString& PedId,
	const FVector& TargetWorldCm,
	const float SpeedCmPerSec,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetTarget(PedId, TargetWorldCm, SpeedCmPerSec))
	{
		OutError = FString::Printf(TEXT("Failed to set target for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ObservePedestrian(const FString& PedId, const FName StartSection, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (StartSection.IsNone())
	{
		if (!PedSubsystem->ExecObserve(PedId))
		{
			OutError = FString::Printf(TEXT("Failed to observe pedestrian '%s'."), *PedId);
			return false;
		}
		return true;
	}

	APedestrianCharacter* Ped = PedSubsystem->FindPedestrian(PedId);
	if (Ped == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to observe pedestrian '%s'."), *PedId);
		return false;
	}

	Ped->CmdPlayObserve(StartSection);
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::PlayPedestrianAnimation(
	const FString& PedId,
	const FString& AnimationAssetPath,
	const FName StartSection,
	const float PlayRate,
	const int32 LoopCount,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	const FString TrimmedAnimationAssetPath = AnimationAssetPath.TrimStartAndEnd();
	if (TrimmedAnimationAssetPath.IsEmpty())
	{
		OutError = TEXT("Animation asset path is required.");
		return false;
	}

	APedestrianCharacter* Ped = PedSubsystem->FindPedestrian(PedId);
	if (Ped == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to find pedestrian '%s'."), *PedId);
		return false;
	}

	UAnimationAsset* AnimationAsset = Cast<UAnimationAsset>(FSoftObjectPath(TrimmedAnimationAssetPath).TryLoad());
	if (AnimationAsset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load animation asset '%s'."), *TrimmedAnimationAssetPath);
		return false;
	}

	if (!Ped->CmdPlayAnimationAsset(AnimationAsset, StartSection, PlayRate, LoopCount))
	{
		OutError = FString::Printf(TEXT("Failed to play animation '%s' for pedestrian '%s'."), *TrimmedAnimationAssetPath, *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::CommitPedestrianCross(
	const FString& PedId,
	const FVector& TargetWorldCm,
	const float SpeedCmPerSec,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecCommitCross(PedId, TargetWorldCm, SpeedCmPerSec))
	{
		OutError = FString::Printf(TEXT("Failed to commit cross for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::StopPedestrian(const FString& PedId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecStop(PedId))
	{
		OutError = FString::Printf(TEXT("Failed to stop pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianVariant(const FString& PedId, const FName VariantId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetVariant(PedId, VariantId))
	{
		OutError = FString::Printf(TEXT("Failed to set variant '%s' for pedestrian '%s'."), *VariantId.ToString(), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ReleasePedestrian(const FString& PedId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecRelease(PedId))
	{
		OutError = FString::Printf(TEXT("Failed to release pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SpawnCrowd(
	const FCrowdSpawnRequest& Request,
	FCrowdSpawnResult& OutResult,
	FString& OutError) const
{
	OutResult = FCrowdSpawnResult();

	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	OutResult = PedSubsystem->SpawnCrowd(Request);
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ClearCrowdGroup(const FName GroupId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ClearCrowdGroup(GroupId))
	{
		OutError = FString::Printf(TEXT("Failed to clear crowd group '%s'."), *GroupId.ToString());
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::RespawnCrowd(
	const FName GroupId,
	const int32 NewSeed,
	FCrowdSpawnResult& OutResult,
	FString& OutError) const
{
	OutResult = FCrowdSpawnResult();

	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	OutResult = PedSubsystem->RespawnCrowd(GroupId, NewSeed);
	return true;
}

AActor* UAeroRuntimeOrchestrationSubsystem::ResolvePedestrianActor(const FString& PedId) const
{
	FString Error;
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(Error);
	return PedSubsystem != nullptr ? PedSubsystem->FindPedestrian(PedId) : nullptr;
}

UPedestrianWorldSubsystem* UAeroRuntimeOrchestrationSubsystem::ResolvePedestrianSubsystem(FString& OutError) const
{
	OutError.Reset();

	UPedestrianWorldSubsystem* PedSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UPedestrianWorldSubsystem>() : nullptr;
	if (PedSubsystem == nullptr)
	{
		OutError = TEXT("PedestrianWorldSubsystem unavailable.");
	}
	return PedSubsystem;
}
