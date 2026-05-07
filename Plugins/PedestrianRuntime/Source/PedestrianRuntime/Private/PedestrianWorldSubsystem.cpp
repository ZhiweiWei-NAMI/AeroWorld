#include "PedestrianWorldSubsystem.h"

#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "GroundPlacementUtils.h"
#include "PedestrianCharacter.h"
#include "PedestrianRuntimeLog.h"
#include "PedestrianRuntimeSettings.h"

namespace
{
FString NormalizePedId(const FString& PedId)
{
	return PedId.TrimStartAndEnd();
}

bool HasAnyAllowedEnumValue(const TArray<ECrowdGender>& AllowedValues, ECrowdGender Value)
{
	return AllowedValues.Num() == 0 || AllowedValues.Contains(Value);
}

bool HasAnyAllowedEnumValue(const TArray<ECrowdAgeGroup>& AllowedValues, ECrowdAgeGroup Value)
{
	return AllowedValues.Num() == 0 || AllowedValues.Contains(Value);
}

bool IsAuthoritativeCrowdVariant(FName VariantId)
{
	return !VariantId.IsNone();
}

void GatherAppearanceTags(const FCrowdAppearanceEntry& Appearance, TSet<FName>& OutTags)
{
	OutTags.Reset();

	for (const FName& Tag : Appearance.SpawnTags)
	{
		if (!Tag.IsNone())
		{
			OutTags.Add(Tag);
		}
	}

	for (const FName& Tag : Appearance.AccessoryTags)
	{
		if (!Tag.IsNone())
		{
			OutTags.Add(Tag);
		}
	}
}

bool ContainsAllTags(const TSet<FName>& SourceTags, const TArray<FName>& RequiredTags)
{
	for (const FName& Tag : RequiredTags)
	{
		if (!Tag.IsNone() && !SourceTags.Contains(Tag))
		{
			return false;
		}
	}

	return true;
}

bool ContainsAnyBlockedTag(const TSet<FName>& SourceTags, const TArray<FName>& BlockedTags)
{
	for (const FName& Tag : BlockedTags)
	{
		if (!Tag.IsNone() && SourceTags.Contains(Tag))
		{
			return true;
		}
	}

	return false;
}

float ResolveScaleValue(const FVector2D& ScaleRange, FRandomStream& Stream)
{
	const float MinScale = FMath::Max(0.01f, FMath::Min(ScaleRange.X, ScaleRange.Y));
	const float MaxScale = FMath::Max(0.01f, FMath::Max(ScaleRange.X, ScaleRange.Y));
	return FMath::IsNearlyEqual(MinScale, MaxScale) ? MinScale : Stream.FRandRange(MinScale, MaxScale);
}

TSubclassOf<APedestrianCharacter> ResolveDefaultPedestrianClass(FString& OutError)
{
	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	if (Settings == nullptr)
	{
		OutError = TEXT("PedestrianRuntimeSettings is unavailable.");
		return nullptr;
	}

	TSubclassOf<APedestrianCharacter> PedestrianClass = Settings->DefaultPedestrianClass.LoadSynchronous();
	if (PedestrianClass == nullptr)
	{
		OutError = TEXT("DefaultPedestrianClass failed to load.");
	}

	return PedestrianClass;
}

FName ResolveDefaultSpawnVariantId()
{
	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	return Settings != nullptr ? Settings->DefaultSpawnVariantId : NAME_None;
}

FName ResolveSpawnVariantId(FName RequestedVariantId)
{
	return RequestedVariantId.IsNone() ? ResolveDefaultSpawnVariantId() : RequestedVariantId;
}

bool TryResolveGroundPlacement(UWorld* World, const FVector& RequestedWorldCm, FVector& OutGroundWorldCm, FVector& OutSurfaceNormalWorld, FString* OutGroundSource = nullptr)
{
	AeroGroundPlacement::FResolvedGroundPlacement Placement;
	if (!AeroGroundPlacement::ResolveGroundPlacement(World, RequestedWorldCm, Placement))
	{
		OutGroundWorldCm = RequestedWorldCm;
		OutSurfaceNormalWorld = FVector::UpVector;
		if (OutGroundSource != nullptr)
		{
			OutGroundSource->Reset();
		}
		return false;
	}

	OutGroundWorldCm = Placement.GroundWorldCm;
	OutSurfaceNormalWorld = Placement.SurfaceNormalWorld;
	if (OutGroundSource != nullptr)
	{
		*OutGroundSource = Placement.Source;
	}
	return true;
}

void AlignPedestrianToGround(APedestrianCharacter* Pedestrian, const FVector& GroundWorldCm, const FVector& SurfaceNormalWorld)
{
	if (!IsValid(Pedestrian))
	{
		return;
	}

	Pedestrian->AlignToGroundPlacement(GroundWorldCm, SurfaceNormalWorld);
}

APedestrianCharacter* SpawnPedestrianActor(
	UWorld* World,
	TSubclassOf<APedestrianCharacter> PedestrianClass,
	const FVector& SpawnLocation,
	float YawDeg,
	ESpawnActorCollisionHandlingMethod CollisionHandling,
	const FString& PedId,
	FName InitialVariantId,
	FString& OutError)
{
	if (World == nullptr)
	{
		OutError = TEXT("No valid world is available for pedestrian spawning.");
		return nullptr;
	}

	if (PedestrianClass == nullptr)
	{
		OutError = TEXT("Pedestrian class is null.");
		return nullptr;
	}

	const FTransform SpawnTransform(FRotator(0.0f, YawDeg, 0.0f), SpawnLocation);
	APedestrianCharacter* SpawnedPedestrian = World->SpawnActorDeferred<APedestrianCharacter>(
		PedestrianClass,
		SpawnTransform,
		nullptr,
		nullptr,
		CollisionHandling);
	if (SpawnedPedestrian == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to spawn pedestrian '%s'."), *PedId);
		return nullptr;
	}

	SpawnedPedestrian->PedId = PedId;
	SpawnedPedestrian->InitialVariantId = InitialVariantId;
	SpawnedPedestrian->FinishSpawning(SpawnTransform);
	return SpawnedPedestrian;
}

FString MakeCrowdPedId(FName GroupId, int32 Index)
{
	const FString Prefix = GroupId.IsNone() ? FString(TEXT("crowd")) : GroupId.ToString();
	return FString::Printf(TEXT("%s.%03d"), *Prefix, Index + 1);
}
} // namespace

void UPedestrianWorldSubsystem::Deinitialize()
{
	TArray<FName> GroupIds;
	CrowdGroups.GetKeys(GroupIds);
	for (const FName GroupId : GroupIds)
	{
		ClearCrowdGroupInternal(GroupId, false);
	}

	PedMap.Reset();
	DynamicPedIds.Reset();
	CrowdGroups.Reset();

	Super::Deinitialize();
}

void UPedestrianWorldSubsystem::RegisterPedestrian(APedestrianCharacter* Ped)
{
	if (!IsValid(Ped))
	{
		return;
	}

	const FString NormalizedPedId = NormalizePedId(Ped->PedId);
	if (NormalizedPedId.IsEmpty())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("RegisterPedestrian skipped: actor '%s' has an empty PedId."), *Ped->GetName());
		return;
	}

	Ped->PedId = NormalizedPedId;
	PedMap.Add(NormalizedPedId, Ped);
}

void UPedestrianWorldSubsystem::UnregisterPedestrian(APedestrianCharacter* Ped)
{
	if (!IsValid(Ped))
	{
		return;
	}

	const FString NormalizedPedId = NormalizePedId(Ped->PedId);
	if (NormalizedPedId.IsEmpty())
	{
		return;
	}

	if (const TWeakObjectPtr<APedestrianCharacter>* ExistingPed = PedMap.Find(NormalizedPedId))
	{
		if (!ExistingPed->IsValid() || ExistingPed->Get() == Ped)
		{
			PedMap.Remove(NormalizedPedId);
		}
	}

	if (DynamicPedIds.Remove(NormalizedPedId) > 0)
	{
		for (TPair<FName, FCrowdGroupState>& Pair : CrowdGroups)
		{
			Pair.Value.SpawnedIds.Remove(NormalizedPedId);
		}
	}
}

APedestrianCharacter* UPedestrianWorldSubsystem::FindPedestrian(const FString& PedId) const
{
	const FString NormalizedPedId = NormalizePedId(PedId);
	if (NormalizedPedId.IsEmpty())
	{
		return nullptr;
	}

	const TWeakObjectPtr<APedestrianCharacter>* ExistingPed = PedMap.Find(NormalizedPedId);
	return ExistingPed != nullptr && ExistingPed->IsValid() ? ExistingPed->Get() : nullptr;
}

bool UPedestrianWorldSubsystem::ExecReset(const FString& PedId, const FVector& Loc, float YawDeg, const bool bUseProvidedGroundPoint)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.reset"));
	if (!IsValid(Ped))
	{
		return false;
	}

	FVector GroundWorldCm = Loc;
	FVector SurfaceNormalWorld = FVector::UpVector;
	FString GroundSource;
	if (!bUseProvidedGroundPoint && !TryResolveGroundPlacement(GetWorld(), Loc, GroundWorldCm, SurfaceNormalWorld, &GroundSource))
	{
		UE_LOG(
			LogPedestrianRuntime,
			Warning,
			TEXT("ped.reset ground projection failed: PedId='%s' requested='%s'."),
			*NormalizePedId(PedId),
			*Loc.ToString());
	}
	else if (!bUseProvidedGroundPoint && !GroundSource.IsEmpty())
	{
		UE_LOG(
			LogPedestrianRuntime,
			Log,
			TEXT("ped.reset grounded: PedId='%s' requested='%s' resolved='%s' source='%s'."),
			*NormalizePedId(PedId),
			*Loc.ToString(),
			*GroundWorldCm.ToString(),
			*GroundSource);
	}

	Ped->CmdReset(GroundWorldCm, YawDeg);
	AlignPedestrianToGround(Ped, GroundWorldCm, SurfaceNormalWorld);
	return true;
}

bool UPedestrianWorldSubsystem::ExecSetFramePose(
	const FString& PedId,
	const FVector& Loc,
	const float YawDeg,
	const bool bWalking,
	const float SpeedCmPerSec,
	const bool bUseProvidedGroundPoint)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.frame_pose"));
	if (!IsValid(Ped))
	{
		return false;
	}

	FVector GroundWorldCm = Loc;
	FVector SurfaceNormalWorld = FVector::UpVector;
	FString GroundSource;
	if (!bUseProvidedGroundPoint && !TryResolveGroundPlacement(GetWorld(), Loc, GroundWorldCm, SurfaceNormalWorld, &GroundSource))
	{
		UE_LOG(
			LogPedestrianRuntime,
			Warning,
			TEXT("ped.frame_pose ground projection failed: PedId='%s' requested='%s'."),
			*NormalizePedId(PedId),
			*Loc.ToString());
	}
	else if (!bUseProvidedGroundPoint && !GroundSource.IsEmpty())
	{
		UE_LOG(
			LogPedestrianRuntime,
			Log,
			TEXT("ped.frame_pose grounded: PedId='%s' requested='%s' resolved='%s' source='%s' walking=%s speed=%.2f."),
			*NormalizePedId(PedId),
			*Loc.ToString(),
			*GroundWorldCm.ToString(),
			*GroundSource,
			bWalking ? TEXT("true") : TEXT("false"),
			SpeedCmPerSec);
	}

	Ped->CmdSetFramePose(GroundWorldCm, YawDeg, bWalking, SpeedCmPerSec);
	AlignPedestrianToGround(Ped, GroundWorldCm, SurfaceNormalWorld);
	return true;
}

bool UPedestrianWorldSubsystem::ExecObserve(const FString& PedId)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.observe"));
	if (!IsValid(Ped))
	{
		return false;
	}

	Ped->CmdPlayObserve();
	return true;
}

bool UPedestrianWorldSubsystem::ExecCommitCross(const FString& PedId, const FVector& Target, float SpeedCmPerSec)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.commit_cross"));
	if (!IsValid(Ped))
	{
		return false;
	}

	Ped->CmdCommitCross(Target, SpeedCmPerSec);
	return true;
}

bool UPedestrianWorldSubsystem::ExecStop(const FString& PedId)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.stop"));
	if (!IsValid(Ped))
	{
		return false;
	}

	Ped->CmdStop();
	return true;
}

bool UPedestrianWorldSubsystem::ExecSetTarget(const FString& PedId, const FVector& Target, float SpeedCmPerSec)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.set_target"));
	if (!IsValid(Ped))
	{
		return false;
	}

	Ped->CmdSetTarget(Target, SpeedCmPerSec);
	return true;
}

bool UPedestrianWorldSubsystem::ExecSetVariant(const FString& PedId, FName VariantId)
{
	APedestrianCharacter* Ped = ResolvePedestrianOrLog(PedId, TEXT("ped.set_variant"));
	if (!IsValid(Ped))
	{
		return false;
	}

	return Ped->ApplyVariant(VariantId);
}

bool UPedestrianWorldSubsystem::ExecSpawn(const FString& PedId, const FVector& Loc, float YawDeg, FName VariantId, const bool bUseProvidedGroundPoint)
{
	UWorld* World = GetWorld();
	if (World == nullptr || !World->IsGameWorld())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.spawn failed: no valid PIE/game world."));
		return false;
	}

	const FString NormalizedPedId = NormalizePedId(PedId);
	if (NormalizedPedId.IsEmpty())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.spawn failed: PedId is empty."));
		return false;
	}

	if (APedestrianCharacter* ExistingPed = FindPedestrian(NormalizedPedId))
	{
		if (!DynamicPedIds.Contains(NormalizedPedId))
		{
			UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.spawn failed: PedId '%s' is already owned by a non-dynamic pedestrian."), *NormalizedPedId);
			return false;
		}

		PedMap.Remove(NormalizedPedId);
		DynamicPedIds.Remove(NormalizedPedId);
		ExistingPed->Destroy();
	}

	FString Error;
	const TSubclassOf<APedestrianCharacter> PedestrianClass = ResolveDefaultPedestrianClass(Error);
	if (PedestrianClass == nullptr)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.spawn failed: %s"), *Error);
		return false;
	}

	const FName ResolvedVariantId = ResolveSpawnVariantId(VariantId);
	FVector GroundWorldCm = Loc;
	FVector SurfaceNormalWorld = FVector::UpVector;
	FString GroundSource;
	if (!bUseProvidedGroundPoint && !TryResolveGroundPlacement(World, Loc, GroundWorldCm, SurfaceNormalWorld, &GroundSource))
	{
		UE_LOG(
			LogPedestrianRuntime,
			Warning,
			TEXT("ped.spawn ground projection failed: PedId='%s' requested='%s'."),
			*NormalizedPedId,
			*Loc.ToString());
	}
	else if (!bUseProvidedGroundPoint && !GroundSource.IsEmpty())
	{
		UE_LOG(
			LogPedestrianRuntime,
			Log,
			TEXT("ped.spawn grounded: PedId='%s' requested='%s' resolved='%s' source='%s'."),
			*NormalizedPedId,
			*Loc.ToString(),
			*GroundWorldCm.ToString(),
			*GroundSource);
	}

	APedestrianCharacter* SpawnedPedestrian = SpawnPedestrianActor(
		World,
		PedestrianClass,
		GroundWorldCm,
		YawDeg,
		ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn,
		NormalizedPedId,
		ResolvedVariantId,
		Error);
	if (!IsValid(SpawnedPedestrian))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.spawn failed: %s"), *Error);
		return false;
	}

	DynamicPedIds.Add(NormalizedPedId);
	if (!ResolvedVariantId.IsNone() && !SpawnedPedestrian->ApplyVariant(ResolvedVariantId))
	{
		PedMap.Remove(NormalizedPedId);
		DynamicPedIds.Remove(NormalizedPedId);
		SpawnedPedestrian->Destroy();
		return false;
	}

	AlignPedestrianToGround(SpawnedPedestrian, GroundWorldCm, SurfaceNormalWorld);
	return true;
}

bool UPedestrianWorldSubsystem::ExecRelease(const FString& PedId)
{
	const FString NormalizedPedId = NormalizePedId(PedId);
	if (NormalizedPedId.IsEmpty())
	{
		return false;
	}

	APedestrianCharacter* Ped = ResolvePedestrianOrLog(NormalizedPedId, TEXT("ped.release"));
	if (!IsValid(Ped) || !DynamicPedIds.Contains(NormalizedPedId))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ped.release failed: PedId '%s' is not a dynamic pedestrian."), *NormalizedPedId);
		return false;
	}

	PedMap.Remove(NormalizedPedId);
	DynamicPedIds.Remove(NormalizedPedId);
	Ped->Destroy();
	return true;
}

FCrowdSpawnResult UPedestrianWorldSubsystem::SpawnCrowd(const FCrowdSpawnRequest& Request)
{
	FCrowdSpawnResult Result;
	Result.GroupId = Request.GroupId;
	Result.Seed = Request.Seed;

	UWorld* World = GetWorld();
	if (World == nullptr || !World->IsGameWorld())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd skipped: no valid PIE/game world."));
		return Result;
	}

	FCrowdSpawnRequest ResolvedRequest;
	FString Error;
	if (!ResolveCrowdAssets(Request, ResolvedRequest, Error))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd failed for group '%s': %s"), *Request.GroupId.ToString(), *Error);
		return Result;
	}

	Result.GroupId = ResolvedRequest.GroupId;

	TArray<FCrowdRuntimeSelection> Candidates;
	if (!BuildCandidateSelections(ResolvedRequest.AppearancePool, ResolvedRequest.RoleProfile, Candidates, Error))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd failed for group '%s': %s"), *ResolvedRequest.GroupId.ToString(), *Error);
		return Result;
	}

	const TSubclassOf<APedestrianCharacter> PedestrianClass = ResolveDefaultPedestrianClass(Error);
	if (PedestrianClass == nullptr)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd failed for group '%s': %s"), *ResolvedRequest.GroupId.ToString(), *Error);
		return Result;
	}

	const int32 EffectiveCount = FMath::Max(
		0,
		ResolvedRequest.RoleProfile != nullptr && ResolvedRequest.RoleProfile->CountOverride >= 0
			? ResolvedRequest.RoleProfile->CountOverride
			: ResolvedRequest.Count);
	if (EffectiveCount <= 0)
	{
		ClearCrowdGroupInternal(ResolvedRequest.GroupId, false);
		return Result;
	}

	ClearCrowdGroupInternal(ResolvedRequest.GroupId, false);

	ResolvedRequest.Count = EffectiveCount;
	FRandomStream Stream(ResolvedRequest.Seed);
	TArray<FVector> ExistingLocations;
	ExistingLocations.Reserve(EffectiveCount);

	FCrowdGroupState GroupState;
	GroupState.Request = ResolvedRequest;
	GroupState.LastSeed = ResolvedRequest.Seed;

	for (int32 Index = 0; Index < EffectiveCount; ++Index)
	{
		FCrowdAppearanceEntry Appearance;
		if (!SelectAppearance(Candidates, Stream, Appearance))
		{
			++Result.SkippedCount;
			continue;
		}

		FVector SpawnLocation = ResolvedRequest.SpawnOrigin;
		if (!SelectSpawnLocation(ResolvedRequest, ResolvedRequest.RoleProfile, ExistingLocations, Stream, SpawnLocation))
		{
			++Result.SkippedCount;
			continue;
		}

		const float SpawnYawDeg = ResolvedRequest.YawPolicy == ECrowdYawPolicy::Fixed
			? ResolvedRequest.FixedYawDeg
			: Stream.FRandRange(0.0f, 360.0f);
		TArray<FCrowdAccessorySpec> SelectedAccessories;
		GatherSelectedAccessories(Appearance, Stream, SelectedAccessories);
		FVector GroundWorldCm = SpawnLocation;
		FVector SurfaceNormalWorld = FVector::UpVector;
		FString GroundSource;
		if (!ResolvedRequest.bUseProvidedGroundPoint)
		{
			TryResolveGroundPlacement(World, SpawnLocation, GroundWorldCm, SurfaceNormalWorld, &GroundSource);
			if (!GroundSource.IsEmpty())
			{
				UE_LOG(
					LogPedestrianRuntime,
					Log,
					TEXT("crowd.spawn grounded: GroupId='%s' requested='%s' resolved='%s' source='%s'."),
					*ResolvedRequest.GroupId.ToString(),
					*SpawnLocation.ToString(),
					*GroundWorldCm.ToString(),
					*GroundSource);
			}
		}

		const FString PedId = MakeCrowdPedId(ResolvedRequest.GroupId, Index);
		APedestrianCharacter* SpawnedPedestrian = SpawnPedestrianActor(
			World,
			PedestrianClass,
			GroundWorldCm,
			SpawnYawDeg,
			ResolvedRequest.CollisionHandling,
			PedId,
			Appearance.VariantId,
			Error);
		if (!IsValid(SpawnedPedestrian))
		{
			UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd skipped pedestrian '%s': %s"), *PedId, *Error);
			++Result.SkippedCount;
			continue;
		}

		if (Appearance.MaterialVariant.IsEmpty())
		{
			Appearance.MaterialVariant = FString::Printf(TEXT("crowd_%d_%d"), ResolvedRequest.Seed, Index);
		}

		const float UniformScale = ResolveScaleValue(Appearance.ScaleRange, Stream);
		if (!SpawnedPedestrian->ApplyCrowdAppearance(Appearance, UniformScale, SelectedAccessories))
		{
			PedMap.Remove(PedId);
			DynamicPedIds.Remove(PedId);
			SpawnedPedestrian->Destroy();
			UE_LOG(LogPedestrianRuntime, Warning, TEXT("SpawnCrowd skipped pedestrian '%s': ApplyCrowdAppearance failed."), *PedId);
			++Result.SkippedCount;
			continue;
		}

		AlignPedestrianToGround(SpawnedPedestrian, GroundWorldCm, SurfaceNormalWorld);
		DynamicPedIds.Add(PedId);
		Result.SpawnedIds.Add(PedId);
		GroupState.SpawnedIds.Add(PedId);
		ExistingLocations.Add(SpawnedPedestrian->GetActorLocation());
	}

	CrowdGroups.Add(ResolvedRequest.GroupId, MoveTemp(GroupState));
	return Result;
}

bool UPedestrianWorldSubsystem::ClearCrowdGroup(FName GroupId)
{
	return ClearCrowdGroupInternal(GroupId, false);
}

FCrowdSpawnResult UPedestrianWorldSubsystem::RespawnCrowd(FName GroupId, int32 NewSeed)
{
	FCrowdSpawnResult Result;
	Result.GroupId = GroupId;
	Result.Seed = NewSeed;

	const FCrowdGroupState* GroupState = CrowdGroups.Find(GroupId);
	if (GroupState == nullptr)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("RespawnCrowd failed: group '%s' is unknown."), *GroupId.ToString());
		return Result;
	}

	FCrowdSpawnRequest Request = GroupState->Request;
	Request.Seed = NewSeed;
	ClearCrowdGroupInternal(GroupId, true);
	return SpawnCrowd(Request);
}

APedestrianCharacter* UPedestrianWorldSubsystem::ResolvePedestrianOrLog(const FString& PedId, const TCHAR* CommandName) const
{
	APedestrianCharacter* Ped = FindPedestrian(PedId);
	if (!IsValid(Ped))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("%s failed: PedId '%s' was not found."), CommandName, *NormalizePedId(PedId));
	}

	return Ped;
}

bool UPedestrianWorldSubsystem::ResolveCrowdAssets(const FCrowdSpawnRequest& InRequest, FCrowdSpawnRequest& OutResolvedRequest, FString& OutError) const
{
	OutResolvedRequest = InRequest;
	if (OutResolvedRequest.GroupId.IsNone())
	{
		OutResolvedRequest.GroupId = FName(TEXT("crowd.default"));
	}

	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	if (Settings == nullptr)
	{
		OutError = TEXT("PedestrianRuntimeSettings is unavailable.");
		return false;
	}

	if (OutResolvedRequest.AppearancePool == nullptr)
	{
		OutResolvedRequest.AppearancePool = Settings->DefaultCrowdAppearancePool.LoadSynchronous();
	}
	if (OutResolvedRequest.AppearancePool == nullptr)
	{
		OutError = TEXT("Crowd appearance pool is null or failed to load.");
		return false;
	}

	if (OutResolvedRequest.RoleProfile == nullptr)
	{
		OutResolvedRequest.RoleProfile = Settings->DefaultCrowdRoleProfile.LoadSynchronous();
	}
	if (OutResolvedRequest.RoleProfile == nullptr)
	{
		OutError = TEXT("Crowd role profile is null or failed to load.");
		return false;
	}

	return true;
}

bool UPedestrianWorldSubsystem::BuildCandidateSelections(
	const UCrowdAppearancePool* AppearancePool,
	const UCrowdRoleProfile* RoleProfile,
	TArray<FCrowdRuntimeSelection>& OutCandidates,
	FString& OutError) const
{
	OutCandidates.Reset();

	if (AppearancePool == nullptr)
	{
		OutError = TEXT("Crowd appearance pool is null.");
		return false;
	}

	TSet<FName> AppearanceTags;
	for (const FCrowdAppearanceEntry& Appearance : AppearancePool->Entries)
	{
		if (Appearance.VariantId.IsNone() || Appearance.Weight <= 0.0f)
		{
			continue;
		}

		if (!IsAuthoritativeCrowdVariant(Appearance.VariantId))
		{
			UE_LOG(
				LogPedestrianRuntime,
				Warning,
				TEXT("Skipping crowd appearance '%s' with unsupported variant '%s'."),
				*Appearance.AppearanceId.ToString(),
				*Appearance.VariantId.ToString());
			continue;
		}

		if (RoleProfile != nullptr)
		{
			if (!HasAnyAllowedEnumValue(RoleProfile->AllowedGenders, Appearance.Gender))
			{
				continue;
			}

			if (!HasAnyAllowedEnumValue(RoleProfile->AllowedAgeGroups, Appearance.AgeGroup))
			{
				continue;
			}
		}

		GatherAppearanceTags(Appearance, AppearanceTags);
		if (RoleProfile != nullptr)
		{
			if (!ContainsAllTags(AppearanceTags, RoleProfile->RequiredTags))
			{
				continue;
			}

			if (ContainsAnyBlockedTag(AppearanceTags, RoleProfile->BlockedTags))
			{
				continue;
			}
		}

		float EffectiveWeight = Appearance.Weight;
		if (RoleProfile != nullptr)
		{
			for (const FCrowdTagWeightMultiplier& Multiplier : RoleProfile->WeightMultipliers)
			{
				if (!Multiplier.Tag.IsNone() && AppearanceTags.Contains(Multiplier.Tag))
				{
					EffectiveWeight *= FMath::Max(0.0f, Multiplier.Multiplier);
				}
			}
		}

		if (EffectiveWeight <= 0.0f)
		{
			continue;
		}

		FCrowdRuntimeSelection& Selection = OutCandidates.AddDefaulted_GetRef();
		Selection.Appearance = Appearance;
		Selection.EffectiveWeight = EffectiveWeight;
	}

	if (OutCandidates.Num() == 0)
	{
		OutError = TEXT("Crowd appearance pool does not contain any entries compatible with the active role profile.");
		return false;
	}

	return true;
}

bool UPedestrianWorldSubsystem::SelectAppearance(
	const TArray<FCrowdRuntimeSelection>& Candidates,
	FRandomStream& Stream,
	FCrowdAppearanceEntry& OutAppearance) const
{
	double TotalWeight = 0.0;
	for (const FCrowdRuntimeSelection& Candidate : Candidates)
	{
		TotalWeight += Candidate.EffectiveWeight;
	}

	if (TotalWeight <= 0.0)
	{
		return false;
	}

	double RandomWeight = Stream.FRandRange(0.0f, static_cast<float>(TotalWeight));
	double AccumulatedWeight = 0.0;
	for (const FCrowdRuntimeSelection& Candidate : Candidates)
	{
		AccumulatedWeight += Candidate.EffectiveWeight;
		if (RandomWeight <= AccumulatedWeight)
		{
			OutAppearance = Candidate.Appearance;
			return true;
		}
	}

	OutAppearance = Candidates.Last().Appearance;
	return true;
}

bool UPedestrianWorldSubsystem::SelectSpawnLocation(
	const FCrowdSpawnRequest& Request,
	const UCrowdRoleProfile* RoleProfile,
	const TArray<FVector>& ExistingLocations,
	FRandomStream& Stream,
	FVector& OutLocation) const
{
	const float MinSpacing = RoleProfile != nullptr ? FMath::Max(0.0f, RoleProfile->DefaultMinSpacing) : 0.0f;
	const float DefaultRadius = RoleProfile != nullptr ? FMath::Max(0.0f, RoleProfile->DefaultSpawnRadius) : 0.0f;
	const bool bUseSpawnBox = !Request.SpawnBoxExtent.IsNearlyZero(1.0f);
	const int32 MaxAttempts = FMath::Max(24, ExistingLocations.Num() * 8 + 24);

	for (int32 Attempt = 0; Attempt < MaxAttempts; ++Attempt)
	{
		FVector CandidateLocation = Request.SpawnOrigin;
		if (bUseSpawnBox)
		{
			CandidateLocation.X += Stream.FRandRange(-Request.SpawnBoxExtent.X, Request.SpawnBoxExtent.X);
			CandidateLocation.Y += Stream.FRandRange(-Request.SpawnBoxExtent.Y, Request.SpawnBoxExtent.Y);
		}
		else if (DefaultRadius > 0.0f)
		{
			const float AngleRadians = Stream.FRandRange(0.0f, 2.0f * PI);
			const float Radius = FMath::Sqrt(Stream.FRandRange(0.0f, 1.0f)) * DefaultRadius;
			CandidateLocation.X += FMath::Cos(AngleRadians) * Radius;
			CandidateLocation.Y += FMath::Sin(AngleRadians) * Radius;
		}

		bool bPassesSpacing = true;
		if (MinSpacing > 0.0f)
		{
			for (const FVector& ExistingLocation : ExistingLocations)
			{
				const FVector Delta2D = FVector(
					CandidateLocation.X - ExistingLocation.X,
					CandidateLocation.Y - ExistingLocation.Y,
					0.0f);
				if (Delta2D.SizeSquared() < FMath::Square(MinSpacing))
				{
					bPassesSpacing = false;
					break;
				}
			}
		}

		if (bPassesSpacing)
		{
			if (!Request.bUseProvidedGroundPoint)
			{
				UWorld* World = GetWorld();
				if (World != nullptr)
				{
					FVector ProjectedWorldCm = CandidateLocation;
					if (AeroGroundPlacement::TryProjectWorldPointToGround(World, CandidateLocation, ProjectedWorldCm))
					{
						CandidateLocation.Z = ProjectedWorldCm.Z;
					}
				}
			}

			OutLocation = CandidateLocation;
			return true;
		}
	}

	return false;
}

void UPedestrianWorldSubsystem::GatherSelectedAccessories(
	const FCrowdAppearanceEntry& Appearance,
	FRandomStream& Stream,
	TArray<FCrowdAccessorySpec>& OutAccessories) const
{
	OutAccessories.Reset();
	if (Appearance.OptionalAccessories.Num() == 0)
	{
		return;
	}

	TSet<FName> AllowedTags;
	for (const FName& Tag : Appearance.AccessoryTags)
	{
		if (!Tag.IsNone())
		{
			AllowedTags.Add(Tag);
		}
	}

	TSet<FName> ProcessedTags;
	for (int32 Index = 0; Index < Appearance.OptionalAccessories.Num(); ++Index)
	{
		const FCrowdAccessorySpec& AccessorySpec = Appearance.OptionalAccessories[Index];
		if (AccessorySpec.Mesh.IsNull())
		{
			continue;
		}

		const FName AccessoryTag = AccessorySpec.AccessoryTag;
		if (!AccessoryTag.IsNone())
		{
			if (ProcessedTags.Contains(AccessoryTag))
			{
				continue;
			}
			ProcessedTags.Add(AccessoryTag);

			if (AllowedTags.Num() > 0 && !AllowedTags.Contains(AccessoryTag))
			{
				continue;
			}

			const float Probability = FMath::Clamp(AccessorySpec.Probability, 0.0f, 1.0f);
			if (Stream.FRand() > Probability)
			{
				continue;
			}

			for (const FCrowdAccessorySpec& GroupedSpec : Appearance.OptionalAccessories)
			{
				if (GroupedSpec.AccessoryTag == AccessoryTag && !GroupedSpec.Mesh.IsNull())
				{
					OutAccessories.Add(GroupedSpec);
				}
			}
			continue;
		}

		const float Probability = FMath::Clamp(AccessorySpec.Probability, 0.0f, 1.0f);
		if (Stream.FRand() <= Probability)
		{
			OutAccessories.Add(AccessorySpec);
		}
	}
}

bool UPedestrianWorldSubsystem::ClearCrowdGroupInternal(FName GroupId, bool bKeepGroupState)
{
	FCrowdGroupState* GroupState = CrowdGroups.Find(GroupId);
	if (GroupState == nullptr)
	{
		return false;
	}

	const TArray<FString> SpawnedIds = GroupState->SpawnedIds;
	for (const FString& SpawnedId : SpawnedIds)
	{
		APedestrianCharacter* Ped = FindPedestrian(SpawnedId);
		PedMap.Remove(SpawnedId);
		DynamicPedIds.Remove(SpawnedId);

		if (IsValid(Ped))
		{
			Ped->Destroy();
		}
	}

	if (bKeepGroupState)
	{
		GroupState->SpawnedIds.Reset();
	}
	else
	{
		CrowdGroups.Remove(GroupId);
	}

	return true;
}
