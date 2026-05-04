#pragma once

#include "CoreMinimal.h"
#include "AeroSemanticTypes.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroAssetPlacementSubsystem.generated.h"

class FJsonObject;
class AActor;

struct FAeroAssetTemplateDefinition
{
	FString LogicalAssetId;
	FString SemanticType;
	FString SpawnBackend;
	FString UEAssetPath;
	FVector DefaultScale = FVector::OneVector;
	double DefaultYawOffsetDeg = 0.0;
	double DefaultZOffsetM = 0.0;
	FString GroundSnapPolicy;
	bool bPhysicsEnabled = false;
	TArray<FString> QueryTags;
	FString AirSimRegistryName;
	bool bIsBlueprint = false;
	FString CollisionProfile;
	FString FeedbackMode;
	FString WorldLayerType;
	FString ZoneKind;
	FString LabelClass;
	bool bRenderRequired = true;
	bool bAnnotationVisible = true;
	bool bReservable = false;
	bool bBlocking = false;
	EAeroMovementMode MovementMode = EAeroMovementMode::Teleport;
	FAeroVisualState DefaultVisualState;
	bool bHasDefaultVisualState = false;
};

struct FAeroAssetInstanceState
{
	FString InstanceId;
	FString LogicalAssetId;
	FVector PositionEnuM = FVector::ZeroVector;
	FRotator RotationDeg = FRotator::ZeroRotator;
	TArray<FString> QueryTags;
	bool bEnabled = true;
	bool bDynamic = false;
	bool bReserved = false;
	FString ReservedBy;
	FString EntityId;
	FString WorldLayerType;
	FString ZoneKind;
	TWeakObjectPtr<AActor> Actor;
	TSharedPtr<FJsonObject> InitialState;
	FString PlacementMode;
	TSharedPtr<FJsonObject> Placement;
	EAeroMovementMode MovementMode = EAeroMovementMode::Teleport;
	FAeroVisualState VisualState;
	bool bHasVisualState = false;
	bool bVisualStateExplicit = false;
	FVector LastResolvedWorldLocationCm = FVector::ZeroVector;
	FString LastGroundSource;
};

UCLASS()
class AEROASSETPLACEMENT_API UAeroAssetPlacementSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;

	void SetMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext);

	bool LoadAssetCatalog(const FString& CatalogPath, FString& OutError);
	bool LoadScenarioObjects(const FString& ScenarioPath, FString& OutError);

	const FAeroAssetTemplateDefinition* FindTemplate(const FString& LogicalAssetId) const;
	const FAeroAssetInstanceState* FindInstance(const FString& InstanceId) const;

	TSharedPtr<FJsonObject> SpawnAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	TSharedPtr<FJsonObject> MoveAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	TSharedPtr<FJsonObject> RemoveAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	TSharedPtr<FJsonObject> ReserveOccupancy(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	TSharedPtr<FJsonObject> ReleaseOccupancy(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	TSharedPtr<FJsonObject> QueryNearest(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const;

	bool SpawnOrUpdateProxy(const FString& InstanceId, const FString& LogicalAssetId, const FVector& PositionEnuM, const FRotator& RotationDeg, const TArray<FString>& QueryTags, const FString& EntityId, const FAeroVisualState* VisualState, FString& OutError);
	bool RemoveProxy(const FString& InstanceId, FString& OutError);

private:
	bool ParseTemplateDefinition(const TSharedPtr<FJsonObject>& Object, FAeroAssetTemplateDefinition& OutTemplate, FString& OutError) const;
	bool ParseScenarioObject(const TSharedPtr<FJsonObject>& Object, FAeroAssetInstanceState& OutInstance, FString& OutError) const;
	bool TryResolvePlacementPosition(const FAeroAssetInstanceState& Instance, FVector& OutPositionEnuM, FRotator& OutRotationDeg) const;
	bool TryBuildTriggerShapeConfig(const FAeroAssetInstanceState& Instance, const FVector& OriginEnuM, FAeroTriggerShapeConfig& OutShapeConfig, FString& OutError) const;
	bool SpawnScenarioActors(FString& OutError);
	bool SpawnActorForInstance(FAeroAssetInstanceState& Instance, FString& OutError);
	void DestroyActorForInstance(FAeroAssetInstanceState& Instance);
	FAeroSemanticBindingData BuildBindingData(const FAeroAssetTemplateDefinition& TemplateDef, const FAeroAssetInstanceState& Instance) const;
	FVector ConvertEnuMetersToWorldCm(const FVector& PositionEnuM) const;
	FVector ConvertWorldCmToEnuMeters(const FVector& WorldLocationCm) const;
	FVector ApplyGroundSnapAndOffsets(const FAeroAssetTemplateDefinition& TemplateDef, FVector WorldLocationCm) const;
	FVector ApplyGroundSnapAndOffsets(const FAeroAssetTemplateDefinition& TemplateDef, FVector WorldLocationCm, FString* OutGroundSource) const;
	bool MoveActorForTemplate(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef, const FVector& WorldLocationCm, const FRotator& FinalRotation) const;
	void AlignActorToGroundByBounds(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef) const;
	void MaybeEmitSweepCollision(AActor* Actor, const FHitResult& SweepHit, const FVector& StartWorldLocationCm) const;
	void ApplyInstanceVisualState(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef, FAeroAssetInstanceState& Instance) const;
	bool TryReadVisualStateField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FAeroVisualState& OutState) const;
	bool TryReadVectorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector, double DefaultZ = 0.0) const;
	bool TryReadRotationField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FRotator& OutRotation) const;
	bool ReadPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionEnuM, FRotator& OutRotation) const;
	bool ReadWorldPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionWorldCm, FRotator& OutRotation) const;
	bool TryResolvePayloadPose(const TSharedPtr<FJsonObject>& Object, FVector& OutPositionEnuM, FVector* OutPositionWorldCm, FRotator& OutRotation) const;
	bool TryReadPolygonArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, TArray<FVector>& OutPolygonEnuM) const;

private:
	FString CurrentMapId;
	FVector CurrentWorldOriginCm = FVector::ZeroVector;
	TMap<FString, FAeroAssetTemplateDefinition> TemplatesById;
	TMap<FString, FAeroAssetInstanceState> InstancesById;
};
