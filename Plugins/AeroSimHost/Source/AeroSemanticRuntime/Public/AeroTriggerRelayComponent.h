#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AeroTriggerRelayComponent.generated.h"

class AActor;
class UPrimitiveComponent;

UCLASS(ClassGroup = (Aero), BlueprintType, Blueprintable, meta = (BlueprintSpawnableComponent))
class AEROSEMANTICRUNTIME_API UAeroTriggerRelayComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAeroTriggerRelayComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	void SetTriggerComponent(UPrimitiveComponent* InTriggerComponent);
	void EmitOverlapEnter(AActor* OtherActor);
	void EmitOverlapExit(AActor* OtherActor);

private:
	UFUNCTION()
	void HandleBeginOverlap(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult);

	UFUNCTION()
	void HandleEndOverlap(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex);

	bool ShouldRelayForActor(AActor* OtherActor) const;
	void RelayOverlapEvent(const FString& Type, AActor* OtherActor);

private:
	UPROPERTY(Transient)
	TObjectPtr<UPrimitiveComponent> TriggerComponent = nullptr;

	UPROPERTY(Transient)
	TSet<TObjectPtr<AActor>> ActiveOverlapActors;
};
