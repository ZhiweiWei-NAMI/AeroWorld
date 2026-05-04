#include "AeroCollisionRelayComponent.h"

#include "AeroFeedbackSubsystem.h"
#include "AeroSemanticRuntimeHelpers.h"
#include "Components/PrimitiveComponent.h"
#include "GameFramework/Actor.h"

UAeroCollisionRelayComponent::UAeroCollisionRelayComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAeroCollisionRelayComponent::BeginPlay()
{
	Super::BeginPlay();
	BindToOwnerPrimitives();
}

void UAeroCollisionRelayComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	for (UPrimitiveComponent* Primitive : BoundPrimitives)
	{
		if (IsValid(Primitive))
		{
			Primitive->OnComponentHit.RemoveDynamic(this, &UAeroCollisionRelayComponent::HandleComponentHit);
		}
	}
	BoundPrimitives.Reset();

	Super::EndPlay(EndPlayReason);
}

void UAeroCollisionRelayComponent::HandleComponentHit(UPrimitiveComponent* HitComponent, AActor* OtherActor, UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit)
{
	AActor* OwnerActor = GetOwner();
	if (!IsValid(OwnerActor) || !IsValid(OtherActor) || OwnerActor == OtherActor)
	{
		return;
	}

	FAeroSemanticBindingData SelfBinding;
	if (!FAeroSemanticRuntimeHelpers::ResolveSemanticBinding(OwnerActor, SelfBinding) || (SelfBinding.FeedbackMode != EAeroFeedbackMode::Hit && SelfBinding.FeedbackMode != EAeroFeedbackMode::Both))
	{
		return;
	}

	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>() : nullptr;
	if (FeedbackSubsystem == nullptr)
	{
		return;
	}

	FAeroFeedbackEvent Event;
	Event.Type = TEXT("collision");
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(OwnerActor, true, Event);
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(OtherActor, false, Event);
	Event.SourceActorId = OwnerActor->GetName();
	Event.OtherActorId = OtherActor->GetName();
	Event.Collision.ContactPointEnuM = FeedbackSubsystem->WorldCmToEnuM(Hit.ImpactPoint);
	Event.Collision.ContactNormalEnu = FVector(Hit.ImpactNormal);
	Event.Collision.RelativeSpeedMps = FVector::Distance(OwnerActor->GetVelocity(), OtherActor->GetVelocity()) / 100.0;
	Event.Collision.Impulse = NormalImpulse.Size();
	Event.Collision.bBlocking = Hit.bBlockingHit;
	FeedbackSubsystem->EnqueueFeedback(MoveTemp(Event));
}

void UAeroCollisionRelayComponent::BindToOwnerPrimitives()
{
	BoundPrimitives.Reset();

	AActor* OwnerActor = GetOwner();
	if (!IsValid(OwnerActor))
	{
		return;
	}

	TArray<UActorComponent*> Components;
	OwnerActor->GetComponents(UPrimitiveComponent::StaticClass(), Components);
	for (UActorComponent* Component : Components)
	{
		UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Component);
		if (!IsValid(Primitive))
		{
			continue;
		}

		Primitive->OnComponentHit.AddDynamic(this, &UAeroCollisionRelayComponent::HandleComponentHit);
		BoundPrimitives.Add(Primitive);
	}
}
