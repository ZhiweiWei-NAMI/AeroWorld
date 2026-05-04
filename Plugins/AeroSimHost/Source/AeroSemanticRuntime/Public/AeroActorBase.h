#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AeroActorBase.generated.h"

class UAeroCollisionRelayComponent;
class UAeroSemanticBindingComponent;
class USceneComponent;

UCLASS(BlueprintType, Blueprintable)
class AEROSEMANTICRUNTIME_API AAeroActorBase : public AActor
{
	GENERATED_BODY()

public:
	AAeroActorBase();

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<USceneComponent> SceneRoot = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroSemanticBindingComponent> SemanticBinding = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroCollisionRelayComponent> CollisionRelay = nullptr;
};
