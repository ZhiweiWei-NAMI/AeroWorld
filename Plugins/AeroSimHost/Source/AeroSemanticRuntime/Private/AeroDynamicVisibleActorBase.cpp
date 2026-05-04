#include "AeroDynamicVisibleActorBase.h"

#include "AeroCollisionRelayComponent.h"
#include "AeroSemanticBindingComponent.h"
#include "Components/BoxComponent.h"
#include "Components/LightComponent.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"

AAeroDynamicVisibleActorBase::AAeroDynamicVisibleActorBase()
{
	PrimaryActorTick.bCanEverTick = false;

	CollisionRoot = CreateDefaultSubobject<UBoxComponent>(TEXT("CollisionRoot"));
	CollisionRoot->SetBoxExtent(CollisionBoxExtentCm);
	CollisionRoot->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	CollisionRoot->SetCollisionProfileName(TEXT("BlockAllDynamic"));
	CollisionRoot->SetSimulatePhysics(false);
	CollisionRoot->SetNotifyRigidBodyCollision(true);
	CollisionRoot->SetGenerateOverlapEvents(false);
	CollisionRoot->SetMobility(EComponentMobility::Movable);
	RootComponent = CollisionRoot;

	VisualRoot = CreateDefaultSubobject<USceneComponent>(TEXT("VisualRoot"));
	VisualRoot->SetupAttachment(CollisionRoot);

	PrimaryMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PrimaryMesh"));
	PrimaryMesh->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(PrimaryMesh);

	MeshSlot01 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot01"));
	MeshSlot01->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot01);

	MeshSlot02 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot02"));
	MeshSlot02->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot02);

	MeshSlot03 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot03"));
	MeshSlot03->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot03);

	MeshSlot04 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot04"));
	MeshSlot04->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot04);

	MeshSlot05 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot05"));
	MeshSlot05->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot05);

	MeshSlot06 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot06"));
	MeshSlot06->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot06);

	MeshSlot07 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot07"));
	MeshSlot07->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot07);

	MeshSlot08 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshSlot08"));
	MeshSlot08->SetupAttachment(VisualRoot);
	ConfigureVisualMesh(MeshSlot08);

	SemanticBinding = CreateDefaultSubobject<UAeroSemanticBindingComponent>(TEXT("SemanticBinding"));
	CollisionRelay = CreateDefaultSubobject<UAeroCollisionRelayComponent>(TEXT("CollisionRelay"));
}

void AAeroDynamicVisibleActorBase::ApplyAeroVisualState_Implementation(const FAeroVisualState& VisualState)
{
	CurrentVisualState = VisualState;

	const bool bShouldHide = VisualState.Mode.Equals(TEXT("hidden"), ESearchCase::IgnoreCase) ||
		VisualState.Mode.Equals(TEXT("invisible"), ESearchCase::IgnoreCase);
	SetActorHiddenInGame(bShouldHide);

	if (VisualState.bHasLightsOn)
	{
		ApplyLightsState(VisualState.bLightsOn);
	}
}

void AAeroDynamicVisibleActorBase::SetCollisionBoxExtentCm(const FVector& InExtentCm)
{
	CollisionBoxExtentCm = FVector(
		FMath::Max(1.0f, InExtentCm.X),
		FMath::Max(1.0f, InExtentCm.Y),
		FMath::Max(1.0f, InExtentCm.Z));

	if (IsValid(CollisionRoot))
	{
		CollisionRoot->SetBoxExtent(CollisionBoxExtentCm);
	}
}

FVector AAeroDynamicVisibleActorBase::GetCollisionBoxExtentCm() const
{
	return CollisionBoxExtentCm;
}

FAeroVisualState AAeroDynamicVisibleActorBase::GetCurrentVisualState() const
{
	return CurrentVisualState;
}

TArray<UStaticMeshComponent*> AAeroDynamicVisibleActorBase::GetMeshSlots() const
{
	TArray<UStaticMeshComponent*> MeshSlots;
	MeshSlots.Reserve(9);
	MeshSlots.Add(PrimaryMesh);
	MeshSlots.Add(MeshSlot01);
	MeshSlots.Add(MeshSlot02);
	MeshSlots.Add(MeshSlot03);
	MeshSlots.Add(MeshSlot04);
	MeshSlots.Add(MeshSlot05);
	MeshSlots.Add(MeshSlot06);
	MeshSlots.Add(MeshSlot07);
	MeshSlots.Add(MeshSlot08);
	return MeshSlots;
}

void AAeroDynamicVisibleActorBase::ConfigureVisualMesh(UStaticMeshComponent* MeshComponent)
{
	if (!IsValid(MeshComponent))
	{
		return;
	}

	MeshComponent->SetMobility(EComponentMobility::Movable);
	MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	MeshComponent->SetGenerateOverlapEvents(false);
	MeshComponent->SetCanEverAffectNavigation(false);
}

void AAeroDynamicVisibleActorBase::ApplyLightsState(bool bEnabled)
{
	TInlineComponentArray<ULightComponent*> LightComponents(this);
	for (ULightComponent* LightComponent : LightComponents)
	{
		if (IsValid(LightComponent))
		{
			LightComponent->SetVisibility(bEnabled);
			LightComponent->SetHiddenInGame(!bEnabled);
		}
	}
}
