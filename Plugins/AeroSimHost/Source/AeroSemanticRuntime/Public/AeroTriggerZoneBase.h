#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AeroSemanticTypes.h"
#include "AeroTriggerZoneBase.generated.h"

class UAeroSemanticBindingComponent;
class UAeroTriggerRelayComponent;
class UBoxComponent;
class UPrimitiveComponent;
class USceneComponent;
class USphereComponent;

UCLASS(BlueprintType, Blueprintable)
class AEROSEMANTICRUNTIME_API AAeroTriggerZoneBase : public AActor
{
	GENERATED_BODY()

public:
	AAeroTriggerZoneBase();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;

	void ConfigureShape(const FAeroTriggerShapeConfig& InShapeConfig);
	bool IsWorldLocationInsideZone(const FVector& WorldLocationCm) const;
	EAeroTriggerShapeKind GetShapeKind() const;
	UPrimitiveComponent* GetActiveTriggerComponent() const;
	const FAeroTriggerShapeConfig& GetShapeConfig() const;

protected:
	void RefreshShapeComponents();
	bool IsLocalPointInsidePolygon(const FVector& LocalPointCm) const;

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<USceneComponent> SceneRoot = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UBoxComponent> BoxTrigger = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<USphereComponent> SphereTrigger = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroSemanticBindingComponent> SemanticBinding = nullptr;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Aero")
	TObjectPtr<UAeroTriggerRelayComponent> TriggerRelay = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Aero")
	FAeroTriggerShapeConfig ShapeConfig;

	UPROPERTY(Transient)
	TSet<TObjectPtr<AActor>> PolygonInsideActors;
};
