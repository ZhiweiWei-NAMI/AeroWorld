#pragma once

#include "CoreMinimal.h"
#include "AeroSemanticTypes.h"

class AActor;
class AAeroTriggerZoneBase;
class UAeroCollisionRelayComponent;
class UAeroSemanticBindingComponent;
class UAeroTriggerRelayComponent;

class AEROSEMANTICRUNTIME_API FAeroSemanticRuntimeHelpers
{
public:
	static UAeroSemanticBindingComponent* EnsureSemanticBinding(AActor* Actor);
	static UAeroCollisionRelayComponent* EnsureCollisionRelay(AActor* Actor);
	static UAeroTriggerRelayComponent* EnsureTriggerRelay(AActor* Actor);
	static void ApplySemanticBinding(AActor* Actor, const FAeroSemanticBindingData& BindingData);
	static bool ApplyVisualState(AActor* Actor, const FAeroVisualState& VisualState);
	static bool ResolveSemanticBinding(const AActor* Actor, FAeroSemanticBindingData& OutBindingData);
	static void CopySemanticBindingToFeedback(const AActor* Actor, bool bAsSource, FAeroFeedbackEvent& InOutEvent);
	static bool ConfigureTriggerActor(AAeroTriggerZoneBase* TriggerActor, const FAeroSemanticBindingData& BindingData, const FAeroTriggerShapeConfig& ShapeConfig);
};
