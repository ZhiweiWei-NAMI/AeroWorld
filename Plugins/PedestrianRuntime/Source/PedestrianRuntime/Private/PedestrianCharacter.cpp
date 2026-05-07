#include "PedestrianCharacter.h"

#include "AeroSemanticRuntimeHelpers.h"
#include "Animation/AnimationAsset.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimMontage.h"
#include "Animation/AnimSequenceBase.h"
#include "Components/CapsuleComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GroundPlacementUtils.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "PedestrianRuntimeLog.h"
#include "PedestrianVariantCatalog.h"
#include "PedestrianWorldSubsystem.h"

namespace
{
FName NormalizeVariantId(FName VariantId)
{
	if (VariantId.IsNone())
	{
		return NAME_None;
	}

	const FString TrimmedId = VariantId.ToString().TrimStartAndEnd();
	if (TrimmedId.IsEmpty())
	{
		return NAME_None;
	}

	return FName(*TrimmedId);
}

bool MontageHasSection(UAnimMontage* Montage, FName SectionName)
{
	if (!IsValid(Montage) || SectionName.IsNone())
	{
		return false;
	}

	return Montage->GetSectionIndex(SectionName) != INDEX_NONE;
}

FVector BuildMeshRelativeLocation(const USkeletalMesh* SkeletalMesh, const float CapsuleHalfHeight, const FVector& AdditionalOffset)
{
	FVector RelativeLocation = AdditionalOffset;
	if (SkeletalMesh == nullptr)
	{
		return RelativeLocation;
	}

	const FBoxSphereBounds MeshBounds = SkeletalMesh->GetBounds();
	const float MeshBottomLocalZ = MeshBounds.Origin.Z - MeshBounds.BoxExtent.Z;
	RelativeLocation.Z += -CapsuleHalfHeight - MeshBottomLocalZ;
	return RelativeLocation;
}
} // namespace

APedestrianCharacter::APedestrianCharacter()
{
	PrimaryActorTick.bCanEverTick = true;
	VariantMeshComponent = GetMesh();
}

void APedestrianCharacter::BeginPlay()
{
	Super::BeginPlay();

	if (!InitialVariantId.IsNone())
	{
		ApplyVariant(InitialVariantId);
	}

	FAeroSemanticBindingData BindingData;
	BindingData.EntityId = PedId.TrimStartAndEnd();
	BindingData.InstanceId = BindingData.EntityId;
	BindingData.LogicalAssetId = TEXT("pedestrian.cityops.basic.v1");
	BindingData.Tags = {TEXT("pedestrian")};
	BindingData.LabelClass = TEXT("pedestrian");
	BindingData.FeedbackMode = EAeroFeedbackMode::Hit;
	FAeroSemanticRuntimeHelpers::ApplySemanticBinding(this, BindingData);
	FAeroSemanticRuntimeHelpers::EnsureCollisionRelay(this);

	if (UWorld* World = GetWorld())
	{
		if (UPedestrianWorldSubsystem* Subsystem = World->GetSubsystem<UPedestrianWorldSubsystem>())
		{
			Subsystem->RegisterPedestrian(this);
		}
	}
}

void APedestrianCharacter::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	if (!bHasMoveTarget)
	{
		return;
	}

	FVector Delta = TargetLocation - GetActorLocation();
	Delta.Z = 0.0f;

	const float DistanceToTarget = Delta.Size();
	if (DistanceToTarget <= FMath::Max(0.0f, AcceptanceRadius))
	{
		CmdStop();
		CurrentState = EPedState::Idle;
		return;
	}

	const FVector MoveDirection = Delta.GetSafeNormal();
	if (MoveDirection.IsNearlyZero())
	{
		return;
	}

	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->MaxWalkSpeed = FMath::Max(0.0f, MoveSpeed);
	}

	AddMovementInput(MoveDirection, 1.0f);

	if (TurnInterpSpeed > KINDA_SMALL_NUMBER)
	{
		FRotator DesiredRotation = MoveDirection.Rotation();
		DesiredRotation.Pitch = 0.0f;
		DesiredRotation.Roll = 0.0f;
		const FRotator NewRotation = FMath::RInterpTo(GetActorRotation(), DesiredRotation, DeltaSeconds, TurnInterpSpeed);
		SetActorRotation(NewRotation);
	}
}

void APedestrianCharacter::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	ClearCrowdAccessories();

	if (UWorld* World = GetWorld())
	{
		if (UPedestrianWorldSubsystem* Subsystem = World->GetSubsystem<UPedestrianWorldSubsystem>())
		{
			Subsystem->UnregisterPedestrian(this);
		}
	}

	Super::EndPlay(EndPlayReason);
}

void APedestrianCharacter::CmdReset(const FVector& Location, float YawDeg)
{
	bHasMoveTarget = false;
	bStartCrossAfterObserve = false;
	TargetLocation = Location;
	PendingCrossTarget = FVector::ZeroVector;
	CurrentState = EPedState::Idle;

	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->StopMovementImmediately();
	}

	StopCurrentMontages(0.1f);

	SetActorLocationAndRotation(Location, FRotator(0.0f, YawDeg, 0.0f));
}

void APedestrianCharacter::CmdSetFramePose(const FVector& Location, float YawDeg, const bool bWalking, const float SpeedCmPerSec)
{
	bHasMoveTarget = false;
	bStartCrossAfterObserve = false;
	TargetLocation = Location;
	PendingCrossTarget = FVector::ZeroVector;
	MoveSpeed = FMath::Max(0.0f, SpeedCmPerSec);
	CurrentState = bWalking ? EPedState::Cross : EPedState::Idle;

	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->StopMovementImmediately();
		MoveComp->MaxWalkSpeed = MoveSpeed;
	}

	if (!bWalking)
	{
		StopCurrentMontages(0.1f);
	}

	SetActorLocationAndRotation(
		Location,
		FRotator(0.0f, YawDeg, 0.0f),
		false,
		nullptr,
		ETeleportType::TeleportPhysics);
}

void APedestrianCharacter::CmdSetTarget(const FVector& InTarget, float InSpeedCmPerSec)
{
	TargetLocation = InTarget;
	MoveSpeed = FMath::Max(0.0f, InSpeedCmPerSec);
	bHasMoveTarget = true;
	CurrentState = EPedState::Cross;

	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->MaxWalkSpeed = MoveSpeed;
	}
}

void APedestrianCharacter::CmdStop()
{
	bHasMoveTarget = false;
	bStartCrossAfterObserve = false;

	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->StopMovementImmediately();
	}

	StopCurrentMontages(0.1f);

	CurrentState = EPedState::Stop;
}

void APedestrianCharacter::CmdPlayObserve(FName StartSection)
{
	CmdStop();
	CurrentState = EPedState::Observe;

	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	if (!IsValid(MeshComp))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayObserve failed: PedId=%s has no mesh component."), *PedId);
		return;
	}

	UAnimInstance* AnimInstance = MeshComp->GetAnimInstance();
	UAnimMontage* ObserveMontageToPlay = ResolveObserveMontage();
	if (!IsValid(AnimInstance) || !IsValid(ObserveMontageToPlay))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayObserve skipped: PedId=%s ObserveMontage missing or no anim instance."), *PedId);
		return;
	}

	const float PlayedLength = AnimInstance->Montage_Play(ObserveMontageToPlay);
	if (PlayedLength <= 0.0f)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayObserve failed: PedId=%s montage did not start."), *PedId);
		return;
	}

	PlayingObserveMontage = ObserveMontageToPlay;

	if (StartSection != NAME_None)
	{
		AnimInstance->Montage_JumpToSection(StartSection, ObserveMontageToPlay);
	}

	FOnMontageEnded EndDelegate;
	EndDelegate.BindUObject(this, &APedestrianCharacter::HandleMontageEnded);
	AnimInstance->Montage_SetEndDelegate(EndDelegate, ObserveMontageToPlay);
}

void APedestrianCharacter::CmdCommitCross(const FVector& InTarget, float InSpeedCmPerSec)
{
	PendingCrossTarget = InTarget;
	MoveSpeed = FMath::Max(0.0f, InSpeedCmPerSec);

	if (CurrentState == EPedState::Observe)
	{
		bStartCrossAfterObserve = true;

		USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
		UAnimInstance* AnimInstance = IsValid(MeshComp) ? MeshComp->GetAnimInstance() : nullptr;
		UAnimMontage* ObserveMontageToPlay = ResolveObserveMontage();
		if (IsValid(AnimInstance) && IsValid(ObserveMontageToPlay) && AnimInstance->Montage_IsPlaying(ObserveMontageToPlay))
		{
			if (MontageHasSection(ObserveMontageToPlay, TEXT("Exit")))
			{
				UE_LOG(LogPedestrianRuntime, Log, TEXT("CmdCommitCross: PedId=%s exiting observe via Exit section."), *PedId);
				AnimInstance->Montage_JumpToSection(TEXT("Exit"), ObserveMontageToPlay);
			}
			else
			{
				// Fall back to stopping observe when the montage has no dedicated exit section.
				UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdCommitCross: PedId=%s observe montage has no Exit section, stopping montage."), *PedId);
				AnimInstance->Montage_Stop(0.1f, ObserveMontageToPlay);
			}
			return;
		}

		bStartCrossAfterObserve = false;
	}

	PlayStartCross();
}

bool APedestrianCharacter::CmdPlayAnimationAsset(UAnimationAsset* AnimationAsset, FName StartSection, float PlayRate, int32 LoopCount)
{
	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	if (!IsValid(MeshComp))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayAnimationAsset failed: PedId=%s has no mesh component."), *PedId);
		return false;
	}

	UAnimInstance* AnimInstance = MeshComp->GetAnimInstance();
	if (!IsValid(AnimInstance) || !IsValid(AnimationAsset))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayAnimationAsset failed: PedId=%s animation asset missing or no anim instance."), *PedId);
		return false;
	}

	CmdStop();
	CurrentState = EPedState::Observe;

	const float SafePlayRate = FMath::Max(0.01f, PlayRate);
	const int32 SafeLoopCount = FMath::Max(1, LoopCount);
	UAnimMontage* MontageToTrack = nullptr;

	if (UAnimMontage* MontageAsset = Cast<UAnimMontage>(AnimationAsset))
	{
		const float PlayedLength = AnimInstance->Montage_Play(MontageAsset, SafePlayRate);
		if (PlayedLength <= 0.0f)
		{
			UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayAnimationAsset failed: PedId=%s montage did not start for asset '%s'."), *PedId, *AnimationAsset->GetPathName());
			return false;
		}

		MontageToTrack = MontageAsset;
		if (StartSection != NAME_None && MontageHasSection(MontageAsset, StartSection))
		{
			AnimInstance->Montage_JumpToSection(StartSection, MontageAsset);
		}
	}
	else if (UAnimSequenceBase* SequenceAsset = Cast<UAnimSequenceBase>(AnimationAsset))
	{
		MontageToTrack = AnimInstance->PlaySlotAnimationAsDynamicMontage(
			SequenceAsset,
			TEXT("DefaultSlot"),
			0.1f,
			0.1f,
			SafePlayRate,
			SafeLoopCount);
		if (!IsValid(MontageToTrack))
		{
			UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayAnimationAsset failed: PedId=%s dynamic montage did not start for asset '%s'."), *PedId, *AnimationAsset->GetPathName());
			return false;
		}
	}
	else
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CmdPlayAnimationAsset failed: PedId=%s unsupported animation asset '%s'."), *PedId, *AnimationAsset->GetPathName());
		return false;
	}

	PlayingObserveMontage = nullptr;
	PlayingStartCrossMontage = nullptr;
	PlayingTransientMontage = MontageToTrack;

	FOnMontageEnded EndDelegate;
	EndDelegate.BindUObject(this, &APedestrianCharacter::HandleMontageEnded);
	AnimInstance->Montage_SetEndDelegate(EndDelegate, MontageToTrack);
	return true;
}

void APedestrianCharacter::PlayStartCross()
{
	CurrentState = EPedState::StartCross;

	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	UAnimInstance* AnimInstance = IsValid(MeshComp) ? MeshComp->GetAnimInstance() : nullptr;
	UAnimMontage* StartCrossMontageToPlay = ResolveStartCrossMontage();
	if (!IsValid(AnimInstance) || !IsValid(StartCrossMontageToPlay))
	{
		CmdSetTarget(PendingCrossTarget, MoveSpeed);
		return;
	}

	const float PlayedLength = AnimInstance->Montage_Play(StartCrossMontageToPlay);
	if (PlayedLength <= 0.0f)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("PlayStartCross failed: PedId=%s montage did not start."), *PedId);
		CmdSetTarget(PendingCrossTarget, MoveSpeed);
		return;
	}

	PlayingStartCrossMontage = StartCrossMontageToPlay;

	FOnMontageEnded EndDelegate;
	EndDelegate.BindUObject(this, &APedestrianCharacter::HandleMontageEnded);
	AnimInstance->Montage_SetEndDelegate(EndDelegate, StartCrossMontageToPlay);
}

void APedestrianCharacter::HandleMontageEnded(UAnimMontage* Montage, bool bInterrupted)
{
	if (Montage == PlayingObserveMontage)
	{
		PlayingObserveMontage = nullptr;

		if (bStartCrossAfterObserve)
		{
			UE_LOG(LogPedestrianRuntime, Log, TEXT("HandleMontageEnded: PedId=%s observe finished, transitioning to start_cross (interrupted=%s)."), *PedId, bInterrupted ? TEXT("true") : TEXT("false"));
			bStartCrossAfterObserve = false;
			PlayStartCross();
			return;
		}

		bStartCrossAfterObserve = false;
		if (CurrentState == EPedState::Observe)
		{
			CurrentState = EPedState::Idle;
		}
		return;
	}

	if (Montage == PlayingStartCrossMontage)
	{
		PlayingStartCrossMontage = nullptr;

		if (!bInterrupted)
		{
			CmdSetTarget(PendingCrossTarget, MoveSpeed);
		}
		else if (CurrentState == EPedState::StartCross)
		{
			CurrentState = EPedState::Idle;
		}
		return;
	}

	if (Montage == PlayingTransientMontage)
	{
		PlayingTransientMontage = nullptr;
		if (CurrentState == EPedState::Observe)
		{
			CurrentState = EPedState::Idle;
		}
	}
}

void APedestrianCharacter::ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState)
{
	CurrentVisualState = VisualState;

	if (!VisualState.VariantId.IsNone())
	{
		ApplyVariant(VisualState.VariantId);
	}

	if (VisualState.Mode.Equals(TEXT("stop"), ESearchCase::IgnoreCase))
	{
		CmdStop();
		return;
	}

	if (VisualState.Mode.Equals(TEXT("idle"), ESearchCase::IgnoreCase))
	{
		CmdStop();
		CurrentState = EPedState::Idle;
		return;
	}

	if (VisualState.Mode.Equals(TEXT("observe"), ESearchCase::IgnoreCase))
	{
		CmdPlayObserve(VisualState.MontageTag);
		return;
	}

	if (VisualState.Mode.Equals(TEXT("start_cross"), ESearchCase::IgnoreCase))
	{
		PlayStartCross();
		return;
	}

	if (!VisualState.MontageTag.IsNone())
	{
		const FString MontageTagValue = VisualState.MontageTag.ToString();
		if (MontageTagValue.Equals(TEXT("observe"), ESearchCase::IgnoreCase))
		{
			CmdPlayObserve();
		}
		else if (MontageTagValue.Equals(TEXT("stop"), ESearchCase::IgnoreCase))
		{
			CmdStop();
		}
		else if (MontageTagValue.Equals(TEXT("start_cross"), ESearchCase::IgnoreCase))
		{
			PlayStartCross();
		}
	}
}

bool APedestrianCharacter::ApplyVariant(FName VariantId)
{
	const FName NormalizedVariantId = NormalizeVariantId(VariantId);
	if (NormalizedVariantId.IsNone())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ApplyVariant failed: PedId=%s VariantId is empty."), *PedId);
		return false;
	}

	if (!IsValid(VariantCatalog))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ApplyVariant failed: PedId=%s VariantCatalog is null."), *PedId);
		return false;
	}

	FPedVariantSpec VariantSpec;
	if (!VariantCatalog->FindVariantById(NormalizedVariantId, VariantSpec))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ApplyVariant failed: PedId=%s VariantId=%s not found."), *PedId, *NormalizedVariantId.ToString());
		return false;
	}
	CurrentGroundContactOffsetCm = VariantSpec.GroundContactOffsetCm;

	bool bApplySuccess = true;
	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	USkeletalMesh* AppliedVariantMesh = nullptr;
	if (!IsValid(MeshComp))
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("ApplyVariant failed: PedId=%s has no target mesh component."), *PedId);
		bApplySuccess = false;
	}
	else if (!VariantSpec.SkeletalMesh.IsNull())
	{
		AppliedVariantMesh = VariantSpec.SkeletalMesh.LoadSynchronous();
		if (IsValid(AppliedVariantMesh))
		{
			MeshComp->SetSkeletalMesh(AppliedVariantMesh);
		}
		else
		{
			UE_LOG(
				LogPedestrianRuntime,
				Warning,
				TEXT("ApplyVariant warning: PedId=%s VariantId=%s mesh load failed."),
				*PedId,
				*NormalizedVariantId.ToString());
			bApplySuccess = false;
		}
	}

	float CapsuleHalfHeight = 0.0f;
	if (VariantSpec.CapsuleRadius > 0.0f && VariantSpec.CapsuleHalfHeight > 0.0f)
	{
		if (UCapsuleComponent* CapsuleComp = GetCapsuleComponent())
		{
			CapsuleComp->SetCapsuleSize(VariantSpec.CapsuleRadius, VariantSpec.CapsuleHalfHeight, true);
			CapsuleHalfHeight = CapsuleComp->GetUnscaledCapsuleHalfHeight();
		}
	}
	else if ((VariantSpec.CapsuleRadius > 0.0f) != (VariantSpec.CapsuleHalfHeight > 0.0f))
	{
		UE_LOG(
			LogPedestrianRuntime,
			Warning,
			TEXT("ApplyVariant warning: PedId=%s VariantId=%s capsule config incomplete, skip capsule update."),
			*PedId,
			*NormalizedVariantId.ToString());
	}
	else if (const UCapsuleComponent* CapsuleComp = GetCapsuleComponent())
	{
		CapsuleHalfHeight = CapsuleComp->GetUnscaledCapsuleHalfHeight();
	}

	if (IsValid(MeshComp))
	{
		if (AppliedVariantMesh == nullptr)
		{
			AppliedVariantMesh = MeshComp->GetSkeletalMeshAsset();
		}

		const FVector CurrentRelativeLocation = MeshComp->GetRelativeLocation();
		FVector TargetRelativeLocation = CurrentRelativeLocation + VariantSpec.MeshRelativeLocationOffset;
		if (AppliedVariantMesh != nullptr)
		{
			TargetRelativeLocation = BuildMeshRelativeLocation(AppliedVariantMesh, CapsuleHalfHeight, VariantSpec.MeshRelativeLocationOffset);
			TargetRelativeLocation.X += CurrentRelativeLocation.X;
			TargetRelativeLocation.Y += CurrentRelativeLocation.Y;
		}
		MeshComp->SetRelativeLocation(TargetRelativeLocation);

		if (!VariantSpec.MeshRelativeRotation.IsZero())
		{
			MeshComp->SetRelativeRotation(VariantSpec.MeshRelativeRotation);
		}

		if (!VariantSpec.MeshRelativeScale.Equals(FVector::OneVector))
		{
			MeshComp->SetRelativeScale3D(VariantSpec.MeshRelativeScale);
		}

		// Keep Character's base mesh offset in sync with runtime variant changes.
		CacheInitialMeshOffset(MeshComp->GetRelativeLocation(), MeshComp->GetRelativeRotation());
	}

	if (VariantSpec.DefaultWalkSpeed > 0.0f)
	{
		MoveSpeed = VariantSpec.DefaultWalkSpeed;
		if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
		{
			MoveComp->MaxWalkSpeed = MoveSpeed;
		}
	}

	ActiveObserveMontage = nullptr;
	if (!VariantSpec.ObserveMontageOverride.IsNull())
	{
		UAnimMontage* ObserveOverride = VariantSpec.ObserveMontageOverride.LoadSynchronous();
		if (IsValid(ObserveOverride))
		{
			ActiveObserveMontage = ObserveOverride;
		}
		else
		{
			UE_LOG(
				LogPedestrianRuntime,
				Warning,
				TEXT("ApplyVariant warning: PedId=%s VariantId=%s ObserveMontageOverride load failed."),
				*PedId,
				*NormalizedVariantId.ToString());
			bApplySuccess = false;
		}
	}

	ActiveStartCrossMontage = nullptr;
	if (!VariantSpec.StartCrossMontageOverride.IsNull())
	{
		UAnimMontage* StartCrossOverride = VariantSpec.StartCrossMontageOverride.LoadSynchronous();
		if (IsValid(StartCrossOverride))
		{
			ActiveStartCrossMontage = StartCrossOverride;
		}
		else
		{
			UE_LOG(
				LogPedestrianRuntime,
				Warning,
				TEXT("ApplyVariant warning: PedId=%s VariantId=%s StartCrossMontageOverride load failed."),
				*PedId,
				*NormalizedVariantId.ToString());
			bApplySuccess = false;
		}
	}

	CurrentVariantId = NormalizedVariantId;

	UE_LOG(
		LogPedestrianRuntime,
		Log,
		TEXT("ApplyVariant: PedId=%s VariantId=%s success=%s"),
		*PedId,
		*CurrentVariantId.ToString(),
		bApplySuccess ? TEXT("true") : TEXT("false"));

	return bApplySuccess;
}

bool APedestrianCharacter::ApplyCrowdAppearance(const FCrowdAppearanceEntry& Appearance, float UniformScale, const TArray<FCrowdAccessorySpec>& AccessoriesToAttach)
{
	const float ClampedScale = FMath::Max(0.01f, UniformScale);
	const bool bVariantApplied = ApplyVariant(Appearance.VariantId);

	SetActorScale3D(FVector(ClampedScale));
	CurrentAppearanceId = Appearance.AppearanceId;
	CurrentMaterialVariant = Appearance.MaterialVariant;

	ApplyMaterialVariant(Appearance.MaterialVariant);
	RefreshCrowdSpawnTags(Appearance.SpawnTags);

	ClearCrowdAccessories();
	for (const FCrowdAccessorySpec& AccessorySpec : AccessoriesToAttach)
	{
		AttachCrowdAccessory(AccessorySpec);
	}

	return bVariantApplied;
}

void APedestrianCharacter::AlignToGroundPlacement(const FVector& GroundWorldCm, const FVector& SurfaceNormalWorld)
{
	const UCapsuleComponent* CapsuleComp = GetCapsuleComponent();
	const float CapsuleHalfHeightCm = CapsuleComp != nullptr ? CapsuleComp->GetScaledCapsuleHalfHeight() : 0.0f;
	FVector PlacementNormalWorld = SurfaceNormalWorld.GetSafeNormal();
	if (PlacementNormalWorld.IsNearlyZero())
	{
		PlacementNormalWorld = FVector::UpVector;
	}

	const float ScaleZ = GetActorScale3D().Z;
	const float ContactOffsetCm = CurrentGroundContactOffsetCm * ScaleZ;
	const float BaseTranslationOffsetZCm = -GetBaseTranslationOffset().Z * ScaleZ;
	const float VisualBelowCapsuleCm = FMath::Max(0.0f, BaseTranslationOffsetZCm - CapsuleHalfHeightCm);
	const FVector PlacementTargetWorldCm = GroundWorldCm + PlacementNormalWorld * (CapsuleHalfHeightCm + ContactOffsetCm + VisualBelowCapsuleCm);
	SetActorLocation(PlacementTargetWorldCm, false, nullptr, ETeleportType::TeleportPhysics);

	FVector BoundsOrigin = FVector::ZeroVector;
	FVector BoundsExtent = FVector::ZeroVector;
	if (USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent())
	{
		const FBoxSphereBounds MeshBounds = MeshComp->Bounds;
		BoundsOrigin = MeshBounds.Origin;
		BoundsExtent = MeshBounds.BoxExtent;
	}
	else
	{
		GetActorBounds(false, BoundsOrigin, BoundsExtent);
	}

	FVector VisibleGroundWorldCm = BoundsOrigin;
	FVector VisibleGroundNormalWorld = FVector::UpVector;
	if (!AeroGroundPlacement::TryProjectWorldPointToGround(GetWorld(), BoundsOrigin, VisibleGroundWorldCm, &VisibleGroundNormalWorld, this))
	{
		return;
	}

	const float BottomZ = BoundsOrigin.Z - BoundsExtent.Z;
	const float DeltaZ = VisibleGroundWorldCm.Z - BottomZ;
	if (FMath::Abs(DeltaZ) <= KINDA_SMALL_NUMBER)
	{
		return;
	}

	AddActorWorldOffset(FVector(0.0f, 0.0f, DeltaZ), false, nullptr, ETeleportType::TeleportPhysics);
}

void APedestrianCharacter::StopCurrentMontages(float BlendOutTime)
{
	PlayingObserveMontage = nullptr;
	PlayingStartCrossMontage = nullptr;
	PlayingTransientMontage = nullptr;

	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	if (!IsValid(MeshComp))
	{
		return;
	}

	if (UAnimInstance* AnimInstance = MeshComp->GetAnimInstance())
	{
		AnimInstance->Montage_Stop(BlendOutTime);
	}
}

UAnimMontage* APedestrianCharacter::ResolveObserveMontage() const
{
	return ActiveObserveMontage ? ActiveObserveMontage : ObserveMontage;
}

UAnimMontage* APedestrianCharacter::ResolveStartCrossMontage() const
{
	return ActiveStartCrossMontage ? ActiveStartCrossMontage : StartCrossMontage;
}

USkeletalMeshComponent* APedestrianCharacter::ResolveVariantMeshComponent()
{
	return IsValid(VariantMeshComponent) ? VariantMeshComponent : GetMesh();
}

void APedestrianCharacter::ClearCrowdAccessories()
{
	for (UStaticMeshComponent* AccessoryComponent : CrowdAccessoryComponents)
	{
		if (IsValid(AccessoryComponent))
		{
			AccessoryComponent->DestroyComponent();
		}
	}
	CrowdAccessoryComponents.Reset();
}

void APedestrianCharacter::RefreshCrowdSpawnTags(const TArray<FName>& SpawnTags)
{
	for (const FName& Tag : AppliedCrowdSpawnTags)
	{
		Tags.Remove(Tag);
	}
	AppliedCrowdSpawnTags.Reset();

	for (const FName& Tag : SpawnTags)
	{
		if (Tag.IsNone())
		{
			continue;
		}

		if (!Tags.Contains(Tag))
		{
			Tags.Add(Tag);
		}
		AppliedCrowdSpawnTags.Add(Tag);
	}
}

void APedestrianCharacter::ApplyMaterialVariant(const FString& MaterialVariant)
{
	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	if (!IsValid(MeshComp))
	{
		return;
	}

	const int32 MaterialCount = MeshComp->GetNumMaterials();
	if (MaterialCount <= 0)
	{
		return;
	}

	float VariantValue = 0.0f;
	if (!MaterialVariant.TrimStartAndEnd().IsEmpty())
	{
		VariantValue = static_cast<float>(GetTypeHash(MaterialVariant) % 1000);
	}

	for (int32 MaterialIndex = 0; MaterialIndex < MaterialCount; ++MaterialIndex)
	{
		UMaterialInstanceDynamic* MID = MeshComp->CreateAndSetMaterialInstanceDynamic(MaterialIndex);
		if (MID == nullptr)
		{
			continue;
		}

		MID->SetScalarParameterValue(TEXT("CrowdVariantEnabled"), MaterialVariant.IsEmpty() ? 0.0f : 1.0f);
		MID->SetScalarParameterValue(TEXT("CrowdVariant"), VariantValue);
	}
}

bool APedestrianCharacter::AttachCrowdAccessory(const FCrowdAccessorySpec& AccessorySpec)
{
	USkeletalMeshComponent* MeshComp = ResolveVariantMeshComponent();
	if (!IsValid(MeshComp) || AccessorySpec.Mesh.IsNull())
	{
		return false;
	}

	UStaticMesh* AccessoryMesh = AccessorySpec.Mesh.LoadSynchronous();
	if (!IsValid(AccessoryMesh))
	{
		UE_LOG(
			LogPedestrianRuntime,
			Warning,
			TEXT("AttachCrowdAccessory skipped: PedId=%s failed to load mesh for tag '%s'."),
			*PedId,
			*AccessorySpec.AccessoryTag.ToString());
		return false;
	}

	UStaticMeshComponent* AccessoryComponent = NewObject<UStaticMeshComponent>(this);
	if (!IsValid(AccessoryComponent))
	{
		return false;
	}

	AccessoryComponent->SetStaticMesh(AccessoryMesh);
	AccessoryComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	AccessoryComponent->SetGenerateOverlapEvents(false);
	AccessoryComponent->SetMobility(EComponentMobility::Movable);
	AccessoryComponent->SetCanEverAffectNavigation(false);
	AccessoryComponent->SetRelativeLocation(AccessorySpec.RelativeLocation);
	AccessoryComponent->SetRelativeRotation(AccessorySpec.RelativeRotation);
	AccessoryComponent->SetRelativeScale3D(AccessorySpec.RelativeScale);

	const FName SocketName = AccessorySpec.SocketName;
	if (!SocketName.IsNone() && MeshComp->DoesSocketExist(SocketName))
	{
		AccessoryComponent->SetupAttachment(MeshComp, SocketName);
	}
	else
	{
		AccessoryComponent->SetupAttachment(MeshComp);
	}

	AddInstanceComponent(AccessoryComponent);
	AccessoryComponent->RegisterComponent();
	CrowdAccessoryComponents.Add(AccessoryComponent);
	return true;
}
