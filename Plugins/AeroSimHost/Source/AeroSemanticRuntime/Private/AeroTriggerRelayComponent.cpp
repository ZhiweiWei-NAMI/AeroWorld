#include "AeroTriggerRelayComponent.h"

#include "AeroFeedbackSubsystem.h"
#include "AeroSemanticRuntimeHelpers.h"
#include "Components/PrimitiveComponent.h"
#include "GameFramework/Actor.h"

UAeroTriggerRelayComponent::UAeroTriggerRelayComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAeroTriggerRelayComponent::BeginPlay()
{
	Super::BeginPlay();

	if (TriggerComponent == nullptr)
	{
		TriggerComponent = Cast<UPrimitiveComponent>(GetOwner() != nullptr ? GetOwner()->GetRootComponent() : nullptr);
	}

	if (IsValid(TriggerComponent))
	{
		TriggerComponent->OnComponentBeginOverlap.AddDynamic(this, &UAeroTriggerRelayComponent::HandleBeginOverlap);
		TriggerComponent->OnComponentEndOverlap.AddDynamic(this, &UAeroTriggerRelayComponent::HandleEndOverlap);
	}
}

void UAeroTriggerRelayComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (IsValid(TriggerComponent))
	{
		TriggerComponent->OnComponentBeginOverlap.RemoveDynamic(this, &UAeroTriggerRelayComponent::HandleBeginOverlap);
		TriggerComponent->OnComponentEndOverlap.RemoveDynamic(this, &UAeroTriggerRelayComponent::HandleEndOverlap);
	}
	ActiveOverlapActors.Reset();

	Super::EndPlay(EndPlayReason);
}

void UAeroTriggerRelayComponent::SetTriggerComponent(UPrimitiveComponent* InTriggerComponent)
{
	if (TriggerComponent == InTriggerComponent)
	{
		return;
	}

	if (IsValid(TriggerComponent))
	{
		TriggerComponent->OnComponentBeginOverlap.RemoveDynamic(this, &UAeroTriggerRelayComponent::HandleBeginOverlap);
		TriggerComponent->OnComponentEndOverlap.RemoveDynamic(this, &UAeroTriggerRelayComponent::HandleEndOverlap);
	}

	TriggerComponent = InTriggerComponent;
	if (IsRegistered() && IsValid(TriggerComponent))
	{
		TriggerComponent->OnComponentBeginOverlap.AddDynamic(this, &UAeroTriggerRelayComponent::HandleBeginOverlap);
		TriggerComponent->OnComponentEndOverlap.AddDynamic(this, &UAeroTriggerRelayComponent::HandleEndOverlap);
	}
}

void UAeroTriggerRelayComponent::EmitOverlapEnter(AActor* OtherActor)
{
	if (!ShouldRelayForActor(OtherActor) || ActiveOverlapActors.Contains(OtherActor))
	{
		return;
	}

	ActiveOverlapActors.Add(OtherActor);
	RelayOverlapEvent(TEXT("overlap_enter"), OtherActor);
}

void UAeroTriggerRelayComponent::EmitOverlapExit(AActor* OtherActor)
{
	if (!IsValid(OtherActor) || !ActiveOverlapActors.Contains(OtherActor))
	{
		return;
	}

	ActiveOverlapActors.Remove(OtherActor);
	RelayOverlapEvent(TEXT("overlap_exit"), OtherActor);
}

void UAeroTriggerRelayComponent::HandleBeginOverlap(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult)
{
	EmitOverlapEnter(OtherActor);
}

void UAeroTriggerRelayComponent::HandleEndOverlap(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex)
{
	EmitOverlapExit(OtherActor);
}

bool UAeroTriggerRelayComponent::ShouldRelayForActor(AActor* OtherActor) const
{
	AActor* OwnerActor = GetOwner();
	if (!IsValid(OwnerActor) || !IsValid(OtherActor) || OwnerActor == OtherActor)
	{
		return false;
	}

	FAeroSemanticBindingData ZoneBinding;
	return FAeroSemanticRuntimeHelpers::ResolveSemanticBinding(OwnerActor, ZoneBinding) && (ZoneBinding.FeedbackMode == EAeroFeedbackMode::Overlap || ZoneBinding.FeedbackMode == EAeroFeedbackMode::Both);
}

void UAeroTriggerRelayComponent::RelayOverlapEvent(const FString& Type, AActor* OtherActor)
{
	AActor* OwnerActor = GetOwner();
	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>() : nullptr;
	if (!IsValid(OwnerActor) || !IsValid(OtherActor) || FeedbackSubsystem == nullptr)
	{
		return;
	}

	FAeroFeedbackEvent Event;
	Event.Type = Type;
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(OtherActor, true, Event);
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(OwnerActor, false, Event);
	Event.SourceActorId = OtherActor->GetName();
	Event.OtherActorId = OwnerActor->GetName();
	FeedbackSubsystem->EnqueueFeedback(MoveTemp(Event));
}
