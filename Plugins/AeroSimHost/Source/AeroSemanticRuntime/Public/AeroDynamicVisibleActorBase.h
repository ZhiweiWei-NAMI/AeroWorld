#pragma once

#include "CoreMinimal.h"
#include "AeroVisualStateReceiver.h"
#include "GameFramework/Actor.h"
#include "AeroDynamicVisibleActorBase.generated.h"

class UBoxComponent;
class UAeroCollisionRelayComponent;
class UAeroSemanticBindingComponent;
class ULightComponent;
class USceneComponent;
class UStaticMeshComponent;

UCLASS(BlueprintType, Blueprintable)
class AEROSEMANTICRUNTIME_API AAeroDynamicVisibleActorBase : public AActor, public IAeroVisualStateReceiver
{
	GENERATED_BODY()

public:
	AAeroDynamicVisibleActorBase();

	virtual void ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState) override;

	UFUNCTION(BlueprintCallable, Category = "Aero|Collision")
	void SetCollisionBoxExtentCm(const FVector& InExtentCm);

	UFUNCTION(BlueprintPure, Category = "Aero|Collision")
	FVector GetCollisionBoxExtentCm() const;

	UFUNCTION(BlueprintPure, Category = "Aero|Visual")
	FAeroVisualState GetCurrentVisualState() const;

	UFUNCTION(BlueprintCallable, Category = "Aero|Visual")
	TArray<UStaticMeshComponent*> GetMeshSlots() const;

protected:
	void ConfigureVisualMesh(UStaticMeshComponent* MeshComponent);
	void ApplyLightsState(bool bEnabled);

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UBoxComponent> CollisionRoot = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<USceneComponent> VisualRoot = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> PrimaryMesh = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot01 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot02 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot03 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot04 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot05 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot06 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot07 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	TObjectPtr<UStaticMeshComponent> MeshSlot08 = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroSemanticBindingComponent> SemanticBinding = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroCollisionRelayComponent> CollisionRelay = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Aero|Collision")
	FVector CollisionBoxExtentCm = FVector(120.0f, 60.0f, 60.0f);

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	FAeroVisualState CurrentVisualState;
};
