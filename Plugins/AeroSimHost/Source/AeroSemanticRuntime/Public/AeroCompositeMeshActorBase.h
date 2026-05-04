#pragma once

#include "CoreMinimal.h"
#include "AeroVisualStateReceiver.h"
#include "GameFramework/Actor.h"
#include "AeroCompositeMeshActorBase.generated.h"

class UAeroCollisionRelayComponent;
class UAeroSemanticBindingComponent;
class USceneComponent;
class UStaticMeshComponent;

UCLASS(BlueprintType, Blueprintable)
class AEROSEMANTICRUNTIME_API AAeroCompositeMeshActorBase : public AActor, public IAeroVisualStateReceiver
{
	GENERATED_BODY()

public:
	AAeroCompositeMeshActorBase();

	virtual void ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState) override;

	UFUNCTION(BlueprintPure, Category = "Aero|Visual")
	FAeroVisualState GetCurrentVisualState() const;

	UFUNCTION(BlueprintCallable, Category = "Aero|Visual")
	TArray<UStaticMeshComponent*> GetMeshSlots() const;

protected:
	void ConfigureVisualMesh(UStaticMeshComponent* MeshComponent);

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<USceneComponent> SceneRoot = nullptr;

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

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroSemanticBindingComponent> SemanticBinding = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroCollisionRelayComponent> CollisionRelay = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero|Visual")
	FAeroVisualState CurrentVisualState;
};
