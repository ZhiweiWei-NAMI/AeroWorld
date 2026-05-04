#include "AeroActorBase.h"

#include "AeroCollisionRelayComponent.h"
#include "AeroSemanticBindingComponent.h"
#include "Components/SceneComponent.h"

AAeroActorBase::AAeroActorBase()
{
	PrimaryActorTick.bCanEverTick = false;

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	RootComponent = SceneRoot;

	SemanticBinding = CreateDefaultSubobject<UAeroSemanticBindingComponent>(TEXT("SemanticBinding"));
	CollisionRelay = CreateDefaultSubobject<UAeroCollisionRelayComponent>(TEXT("CollisionRelay"));
}
