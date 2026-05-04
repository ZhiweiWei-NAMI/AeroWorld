#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AeroCollisionRelayComponent.generated.h"

class UPrimitiveComponent;
class AActor;

UCLASS(ClassGroup = (Aero), BlueprintType, Blueprintable, meta = (BlueprintSpawnableComponent))
class AEROSEMANTICRUNTIME_API UAeroCollisionRelayComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAeroCollisionRelayComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	UFUNCTION()
	void HandleComponentHit(UPrimitiveComponent* HitComponent, AActor* OtherActor, UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit);

	void BindToOwnerPrimitives();

private:
	UPROPERTY(Transient)
	TArray<TObjectPtr<UPrimitiveComponent>> BoundPrimitives;
};
