#include "AeroSemanticRuntimeHelpers.h"

#include "AeroCollisionRelayComponent.h"
#include "AeroSemanticBindingComponent.h"
#include "AeroTriggerRelayComponent.h"
#include "AeroTriggerZoneBase.h"
#include "AeroVisualStateReceiver.h"
#include "GameFramework/Actor.h"

namespace
{
template <typename TComponentType>
TComponentType* EnsureActorComponent(AActor* Actor, const FName ComponentName)
{
	if (!IsValid(Actor))
	{
		return nullptr;
	}

	if (TComponentType* ExistingComponent = Actor->FindComponentByClass<TComponentType>())
	{
		return ExistingComponent;
	}

	TComponentType* NewComponent = NewObject<TComponentType>(Actor, ComponentName);
	if (!IsValid(NewComponent))
	{
		return nullptr;
	}

	Actor->AddInstanceComponent(NewComponent);
	NewComponent->RegisterComponent();
	return NewComponent;
}
}

UAeroSemanticBindingComponent* FAeroSemanticRuntimeHelpers::EnsureSemanticBinding(AActor* Actor)
{
	return EnsureActorComponent<UAeroSemanticBindingComponent>(Actor, TEXT("SemanticBinding"));
}

UAeroCollisionRelayComponent* FAeroSemanticRuntimeHelpers::EnsureCollisionRelay(AActor* Actor)
{
	return EnsureActorComponent<UAeroCollisionRelayComponent>(Actor, TEXT("CollisionRelay"));
}

UAeroTriggerRelayComponent* FAeroSemanticRuntimeHelpers::EnsureTriggerRelay(AActor* Actor)
{
	return EnsureActorComponent<UAeroTriggerRelayComponent>(Actor, TEXT("TriggerRelay"));
}

void FAeroSemanticRuntimeHelpers::ApplySemanticBinding(AActor* Actor, const FAeroSemanticBindingData& BindingData)
{
	if (UAeroSemanticBindingComponent* BindingComponent = EnsureSemanticBinding(Actor))
	{
		BindingComponent->ConfigureFromData(BindingData);
	}
}

bool FAeroSemanticRuntimeHelpers::ApplyVisualState(AActor* Actor, const FAeroVisualState& VisualState)
{
	if (!IsValid(Actor) || !Actor->GetClass()->ImplementsInterface(UAeroVisualStateReceiver::StaticClass()))
	{
		return false;
	}

	IAeroVisualStateReceiver::Execute_ApplyAeroVisualState(Actor, VisualState);
	return true;
}

bool FAeroSemanticRuntimeHelpers::ResolveSemanticBinding(const AActor* Actor, FAeroSemanticBindingData& OutBindingData)
{
	if (!IsValid(Actor))
	{
		return false;
	}

	const UAeroSemanticBindingComponent* BindingComponent = Actor->FindComponentByClass<UAeroSemanticBindingComponent>();
	if (!IsValid(BindingComponent))
	{
		return false;
	}

	OutBindingData = BindingComponent->MakeBindingData();
	if (OutBindingData.EntityId.TrimStartAndEnd().IsEmpty())
	{
		OutBindingData.EntityId = BindingComponent->GetStableEntityId();
	}
	return true;
}

void FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(const AActor* Actor, bool bAsSource, FAeroFeedbackEvent& InOutEvent)
{
	FAeroSemanticBindingData BindingData;
	if (ResolveSemanticBinding(Actor, BindingData))
	{
		const FString StableEntityId = BindingData.EntityId.TrimStartAndEnd().IsEmpty() ? BindingData.InstanceId : BindingData.EntityId;
		if (bAsSource)
		{
			InOutEvent.SourceEntityId = StableEntityId;
			InOutEvent.SourceLogicalAssetId = BindingData.LogicalAssetId;
			InOutEvent.SourceTags = BindingData.Tags;
		}
		else
		{
			InOutEvent.OtherEntityId = StableEntityId;
			InOutEvent.OtherLogicalAssetId = BindingData.LogicalAssetId;
			InOutEvent.OtherTags = BindingData.Tags;
			if (InOutEvent.Overlap.WorldLayerType.IsEmpty())
			{
				InOutEvent.Overlap.WorldLayerType = BindingData.WorldLayerType;
			}
			if (InOutEvent.Overlap.ZoneKind.IsEmpty())
			{
				InOutEvent.Overlap.ZoneKind = BindingData.ZoneKind;
			}
		}
		return;
	}

	if (!IsValid(Actor))
	{
		return;
	}

	if (bAsSource)
	{
		InOutEvent.SourceActorId = Actor->GetName();
	}
	else
	{
		InOutEvent.OtherActorId = Actor->GetName();
	}
}

bool FAeroSemanticRuntimeHelpers::ConfigureTriggerActor(AAeroTriggerZoneBase* TriggerActor, const FAeroSemanticBindingData& BindingData, const FAeroTriggerShapeConfig& ShapeConfig)
{
	if (!IsValid(TriggerActor))
	{
		return false;
	}

	ApplySemanticBinding(TriggerActor, BindingData);
	TriggerActor->ConfigureShape(ShapeConfig);
	return true;
}
