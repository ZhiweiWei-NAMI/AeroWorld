#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroSemanticTypes.h"
#include "SumoTypes.h"
#include "CityRuntimeValidationSubsystem.generated.h"

class AActor;
class FJsonObject;
class IInputProcessor;
class SCityRuntimeValidationPanel;
class SWidget;

enum class ECityValidationStepState : uint8
{
	Idle,
	Running,
	Unavailable,
	Passed,
	Failed
};

enum class ECityTrackedObjectKind : uint8
{
	Pedestrian,
	Crowd,
	Asset,
	RuntimeVehicle
};

struct FCityValidationStepResult
{
	FString Name;
	ECityValidationStepState State = ECityValidationStepState::Idle;
	FString Message;
};

struct FCityTrackedRuntimeObject
{
	FString Id;
	FString LogicalAssetId;
	FString EntityId;
	FString GroupId;
	ECityTrackedObjectKind Kind = ECityTrackedObjectKind::Asset;
	TWeakObjectPtr<AActor> Actor;
	bool bGrounded = false;
	bool bHasFeedback = false;
	FString LastFeedbackType;
	FString GroundingMessage;
};

struct FCityScheduledAction
{
	double ExecuteAt = 0.0;
	TFunction<void()> Action;
};

UCLASS()
class UCityRuntimeValidationSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override;

	void LoadContext();
	void RunFullDemo();
	void ClearDemo();
	void PollFeedbackNow();
	void RecheckGrounding();
	void SpawnPed();
	void ObservePed();
	void CommitCross();
	void SpawnCrowd();
	void SpawnCone();
	void SpawnStreetLightPlaceholder();
	void SpawnSceneVehicle();
	void SpawnRuntimeUAV();
	void MoveSceneVehicle();
	void MoveRuntimeUAV();
	void RemoveAll();
	void PlayPedAnimation(const FString& AnimationAssetPath, const FString& Label);
	void TogglePanelVisibility();

	FString GetHeaderText() const;
	FSlateColor GetHeaderColor() const;
	FString GetStatusSummaryText() const;
	FString GetCapabilitiesText() const;
	FString GetStepResultsText() const;
	FString GetTrackedCountsText() const;
	FString GetFeedbackText() const;
	FString GetPassedObjectsText() const;
	FString GetPendingObjectsText() const;
	FString GetFailedObjectsText() const;

private:
	void EnsurePanel();
	void RemovePanel();
	void RegisterInputPreProcessor();
	void UnregisterInputPreProcessor();
	void ResetValidationState();
	void ResetStepResults();
	void ResetDemoScheduling();
	void ScheduleAction(double DelaySeconds, TFunction<void()> Action);
	void ExecuteScheduledActions();
	void PollActiveHudUavMove();
	void UpdateTrackedObjects();
	void UpdateTrackedObjectActor(FCityTrackedRuntimeObject& TrackedObject);
	void UpdateGroundingState(FCityTrackedRuntimeObject& TrackedObject);
	void RefreshFeedbackLinks();
	void SetOverallState(ECityValidationStepState NewState, const FString& NewErrorMessage = FString());
	void SetStepState(const FString& StepName, ECityValidationStepState NewState, const FString& Message = FString());
	bool ResolveDemoRoadAnchor(FTransform& OutRoadAnchorWorld, FSumoNearestLaneSample* OutRoadSample = nullptr);
	FVector GetDemoOriginWorldCm() const;
	FVector GetCurrentWorldOriginCm() const;
	FVector SnapPointToGround(const FVector& WorldLocationCm) const;
	bool TryFindNearestPreferredGroundWorldCm(const FVector& OriginWorldCm, float SearchRadiusCm, float StepCm, FVector& OutWorldCm) const;
	FVector FindNearestPreferredGroundWorldCm(const FVector& OriginWorldCm, float SearchRadiusCm, float StepCm) const;
	bool QueryNearestTaggedInstanceWorldCm(const FString& QueryTag, const FVector& NearWorldCm, FVector& OutWorldCm) const;
	FVector WorldCmToEnuM(const FVector& WorldLocationCm) const;
	bool CallBridge(const FString& OperationName, const TSharedPtr<FJsonObject>& Payload, TSharedPtr<FJsonObject>& OutPayload, FString& OutError) const;
	bool ApplySceneFrameInternal(const TSharedPtr<FJsonObject>& FramePayload);
	bool LoadContextInternal(const FString& MapId);
	bool SpawnAssetInternal(const FString& AssetId, const FString& LogicalAssetId, const FVector& WorldLocationCm, float YawDeg, bool bSnapToGround = true, FString* OutResolvedAssetId = nullptr);
	bool MoveAssetInternal(const FString& AssetId, const FVector& WorldLocationCm, float YawDeg, bool bSnapToGround = true);
	bool RemoveAssetInternal(const FString& AssetId);
	bool SpawnSceneVehicleInternal(const FString& EntityId, const FString& ProxyTemplateId, const FVector& WorldLocationCm, float YawDeg);
	bool MoveSceneVehicleInternal(const FString& EntityId, const FVector& WorldLocationCm, float YawDeg);
	bool RemoveSceneEntityInternal(const FString& EntityId);
	bool SpawnPedInternal(const FString& PedId, const FVector& WorldLocationCm, float YawDeg, const FName& VariantId = NAME_None, bool bUseProvidedGroundPoint = false);
	bool SetPedVariantInternal(const FString& PedId, const FName& VariantId);
	bool ObservePedInternal(const FString& PedId);
	bool CommitCrossInternal(const FString& PedId, const FVector& TargetWorldCm, float SpeedCmPerSec, bool bSnapToGround = true);
	bool ReleasePedInternal(const FString& PedId);
	bool PlayPedAnimationInternal(const FString& PedId, const FString& AnimationAssetPath);
	bool SpawnCrowdInternal(const FString& GroupId, int32 Count, const FVector& WorldOriginCm, bool bUseProvidedGroundPoint = false);
	bool ClearCrowdInternal(const FString& GroupId);
	AActor* ResolvePedestrianActor(const FString& PedId) const;
	AActor* ResolveAssetActor(const FString& AssetId) const;
	AActor* ResolveRuntimeVehicleActor(const FString& VehicleName) const;
	void TrackPedestrian(const FString& PedId, ECityTrackedObjectKind Kind, const FString& GroupId = FString());
	void TrackAsset(const FString& AssetId, const FString& LogicalAssetId);
	void TrackRuntimeVehicle(const FString& VehicleName, const FString& LogicalAssetId);
	void UntrackGroup(const FString& GroupId);
	FString BuildFeedbackSummary(const FAeroFeedbackEvent& Event) const;
	bool GetDemoReferenceWorldCm(FVector& OutWorldCm) const;
	void SetStepUnavailable(const FString& StepName, const FString& Message);
	bool BindHudRuntimeVehicle(AActor* Actor, const FString& EntityId, const FString& LogicalAssetId, const TArray<FString>& Tags, const FString& LabelClass);

private:
	TSharedPtr<SCityRuntimeValidationPanel> PanelWidget;
	TSharedPtr<SWidget> PanelContainerWidget;
	TSharedPtr<IInputProcessor> InputProcessor;
	TArray<FCityValidationStepResult> StepResults;
	TArray<FCityTrackedRuntimeObject> TrackedObjects;
	TArray<FCityScheduledAction> ScheduledActions;
	TArray<FAeroFeedbackEvent> RecentFeedbackEvents;
	ECityValidationStepState OverallState = ECityValidationStepState::Idle;
	FString CurrentMapId;
	FString LastErrorMessage;
	bool bPanelVisible = false;
	bool bPanelAttached = false;
	bool bDemoRunning = false;
	FString ActiveHudVehicleName;
	FString ActiveHudUavName;
	int32 HudUavSpawnSequence = 0;
};
