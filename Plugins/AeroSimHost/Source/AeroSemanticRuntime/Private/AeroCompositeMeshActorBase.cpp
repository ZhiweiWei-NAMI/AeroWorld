#include "AeroCompositeMeshActorBase.h"

#include "AeroCollisionRelayComponent.h"
#include "AeroSemanticBindingComponent.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"

AAeroCompositeMeshActorBase::AAeroCompositeMeshActorBase()
{
	PrimaryActorTick.bCanEverTick = false;

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	RootComponent = SceneRoot;

	PrimaryMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PrimaryMesh"));
	PrimaryMesh->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(PrimaryMesh);

	MeshSlot01 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot01"));
	MeshSlot01->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(MeshSlot01);

	MeshSlot02 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot02"));
	MeshSlot02->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(MeshSlot02);

	MeshSlot03 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot03"));
	MeshSlot03->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(MeshSlot03);

	MeshSlot04 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot04"));
	MeshSlot04->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(MeshSlot04);

	MeshSlot05 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot05"));
	MeshSlot05->SetupAttachment(SceneRoot);
	ConfigureVisualMesh(MeshSlot05);

	SemanticBinding = CreateDefaultSubobject<UAeroSemanticBindingComponent>(TEXT("SemanticBinding"));
	CollisionRelay = CreateDefaultSubobject<UAeroCollisionRelayComponent>(TEXT("CollisionRelay"));
}

void AAeroCompositeMeshActorBase::ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState)
{
	CurrentVisualState = VisualState;

	const bool bShouldHide = VisualState.Mode.Equals(TEXT("hidden"), ESearchCase::IgnoreCase) ||
		VisualState.Mode.Equals(TEXT("invisible"), ESearchCase::IgnoreCase);
	SetActorHiddenInGame(bShouldHide);
}

FAeroVisualState AAeroCompositeMeshActorBase::GetCurrentVisualState() const
{
	return CurrentVisualState;
}

TArray<UStaticMeshComponent*> AAeroCompositeMeshActorBase::GetMeshSlots() const
{
	TArray<UStaticMeshComponent*> MeshSlots;
	MeshSlots.Reserve(6);
	MeshSlots.Add(PrimaryMesh);
	MeshSlots.Add(MeshSlot01);
	MeshSlots.Add(MeshSlot02);
	MeshSlots.Add(MeshSlot03);
	MeshSlots.Add(MeshSlot04);
	MeshSlots.Add(MeshSlot05);
	return MeshSlots;
}

void AAeroCompositeMeshActorBase::ConfigureVisualMesh(UStaticMeshComponent* MeshComponent)
{
	if (!IsValid(MeshComponent))
	{
		return;
	}

	MeshComponent->SetMobility(EComponentMobility::Movable);
	MeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	MeshComponent->SetCollisionProfileName(TEXT("BlockAllDynamic"));
	MeshComponent->SetGenerateOverlapEvents(false);
	MeshComponent->SetCanEverAffectNavigation(false);
}
