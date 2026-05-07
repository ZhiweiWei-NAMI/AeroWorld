#pragma once

#include "AeroVisualStateReceiver.h"
#include "CrowdTypes.h"
#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "PedestrianCharacter.generated.h"

class UAnimationAsset;
class UAnimMontage;
class UPedestrianVariantCatalog;
class USkeletalMeshComponent;
class UStaticMeshComponent;

UENUM(BlueprintType)
enum class EPedState : uint8
{
	Idle,
	Observe,
	StartCross,
	Cross,
	Stop
};

UCLASS(BlueprintType, Blueprintable)
class PEDESTRIANRUNTIME_API APedestrianCharacter : public ACharacter, public IAeroVisualStateReceiver
{
	GENERATED_BODY()

public:
	APedestrianCharacter();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
	FString PedId = TEXT("PED_0001");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
	float AcceptanceRadius = 40.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
	float MoveSpeed = 140.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
	float TurnInterpSpeed = 6.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Variant")
	TObjectPtr<UPedestrianVariantCatalog> VariantCatalog = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Variant")
	FName InitialVariantId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Variant")
	TObjectPtr<USkeletalMeshComponent> VariantMeshComponent = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	FName CurrentVariantId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Animation")
	TObjectPtr<UAnimMontage> ObserveMontage = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Animation")
	TObjectPtr<UAnimMontage> StartCrossMontage = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pedestrian")
	EPedState CurrentState = EPedState::Idle;

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdReset(const FVector& Location, float YawDeg);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdSetFramePose(const FVector& Location, float YawDeg, bool bWalking, float SpeedCmPerSec);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdSetTarget(const FVector& InTarget, float InSpeedCmPerSec);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdStop();

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdPlayObserve(FName StartSection = NAME_None);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	void CmdCommitCross(const FVector& InTarget, float InSpeedCmPerSec);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Command")
	bool CmdPlayAnimationAsset(UAnimationAsset* AnimationAsset, FName StartSection = NAME_None, float PlayRate = 1.0f, int32 LoopCount = 1);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Variant")
	bool ApplyVariant(FName VariantId);

	bool ApplyCrowdAppearance(const FCrowdAppearanceEntry& Appearance, float UniformScale, const TArray<FCrowdAccessorySpec>& AccessoriesToAttach);
	float GetGroundContactOffsetCm() const { return CurrentGroundContactOffsetCm; }
	void AlignToGroundPlacement(const FVector& GroundWorldCm, const FVector& SurfaceNormalWorld);

	virtual void ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState) override;

private:
	FVector TargetLocation = FVector::ZeroVector;
	FVector PendingCrossTarget = FVector::ZeroVector;
	bool bHasMoveTarget = false;
	bool bStartCrossAfterObserve = false;
	FAeroVisualState CurrentVisualState;

	UPROPERTY(Transient)
	TObjectPtr<UAnimMontage> ActiveObserveMontage = nullptr;

	UPROPERTY(Transient)
	TObjectPtr<UAnimMontage> ActiveStartCrossMontage = nullptr;

	UPROPERTY(Transient)
	TObjectPtr<UAnimMontage> PlayingObserveMontage = nullptr;

	UPROPERTY(Transient)
	TObjectPtr<UAnimMontage> PlayingStartCrossMontage = nullptr;

	UPROPERTY(Transient)
	TObjectPtr<UAnimMontage> PlayingTransientMontage = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant", meta = (AllowPrivateAccess = "true"))
	FName CurrentAppearanceId = NAME_None;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant", meta = (AllowPrivateAccess = "true"))
	FString CurrentMaterialVariant;

	UPROPERTY(Transient)
	TArray<TObjectPtr<UStaticMeshComponent>> CrowdAccessoryComponents;

	UPROPERTY(Transient)
	TArray<FName> AppliedCrowdSpawnTags;

	float CurrentGroundContactOffsetCm = 0.0f;

	void PlayStartCross();
	void HandleMontageEnded(UAnimMontage* Montage, bool bInterrupted);
	void StopCurrentMontages(float BlendOutTime);
	UAnimMontage* ResolveObserveMontage() const;
	UAnimMontage* ResolveStartCrossMontage() const;
	USkeletalMeshComponent* ResolveVariantMeshComponent();
	void ClearCrowdAccessories();
	void RefreshCrowdSpawnTags(const TArray<FName>& SpawnTags);
	void ApplyMaterialVariant(const FString& MaterialVariant);
	bool AttachCrowdAccessory(const FCrowdAccessorySpec& AccessorySpec);
};
