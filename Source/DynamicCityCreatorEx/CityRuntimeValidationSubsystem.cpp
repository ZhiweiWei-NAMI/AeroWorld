#include "CityRuntimeValidationSubsystem.h"

#include "AeroAssetPlacementSubsystem.h"
#include "AeroBridgeWorldSubsystem.h"
#include "AeroFeedbackSubsystem.h"
#include "AeroRuntimeOrchestrationSubsystem.h"
#include "AeroSemanticRuntimeHelpers.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "HAL/IConsoleManager.h"
#include "Framework/Application/IInputProcessor.h"
#include "Framework/Application/SlateApplication.h"
#include "GroundPlacementUtils.h"
#include "InputCoreTypes.h"
#include "SCityRuntimeValidationPanel.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "SumoRoadTopologyQuery.h"
#include "UnrealClient.h"
#include "Widgets/SWeakWidget.h"

namespace
{
constexpr TCHAR* DemoMapId = TEXT("donghu_road_topo");
constexpr TCHAR* DemoPedId = TEXT("hud.ped.demo");
constexpr TCHAR* DemoCrowdGroupId = TEXT("hud.crowd.demo");
constexpr TCHAR* DemoConeAssetId = TEXT("hud.asset.cone");
constexpr TCHAR* DemoStreetLightAssetId = TEXT("hud.asset.streetlight");
constexpr TCHAR* DemoVehicleAssetId = TEXT("hud.asset.vehicle");
constexpr TCHAR* DemoUavNamePrefix = TEXT("hud.uav.demo");
constexpr TCHAR* DemoVehicleLogicalAssetId = TEXT("vehicle.emergency.suv.v1");
constexpr TCHAR* DemoUavRuntimeLogicalAssetId = TEXT("airsim.uav.hud.v1");
constexpr TCHAR* DemoConeLogicalAssetId = TEXT("prop.roadwork.traffic_cone.v1");
constexpr TCHAR* DemoStreetLightLogicalAssetId = TEXT("prop.traffic_control.signal_light.v1");
constexpr TCHAR* DemoPedVariantId = TEXT("adult_female_commuter");
constexpr TCHAR* StepSpawnVehicle = TEXT("Spawn Vehicle Proxy");
constexpr TCHAR* StepMoveVehicle = TEXT("Move Vehicle Proxy");
constexpr float GroundingToleranceCm = 18.0f;
constexpr float DemoGroundSearchRadiusCm = 1500.0f;
constexpr float DemoGroundSearchStepCm = 150.0f;
constexpr int32 MaxRecentFeedbackEvents = 10;
const FVector DemoPedSpawnLocalOffsetCm(0.0f, 220.0f, 0.0f);
const FVector DemoPedCrossTargetLocalOffsetCm(260.0f, 220.0f, 0.0f);
const FVector DemoCrowdSpawnLocalOffsetCm(120.0f, -180.0f, 0.0f);
const FVector DemoConeSpawnLocalOffsetCm(520.0f, 0.0f, 0.0f);
const FVector DemoStreetLightSpawnLocalOffsetCm(760.0f, 140.0f, 0.0f);
const FVector DemoVehicleSpawnLocalOffsetCm(0.0f, -260.0f, 0.0f);
const FVector DemoVehicleMoveLocalOffsetCm(3000.0f, 0.0f, 0.0f);
const FVector DemoUavSpawnLocalOffsetCm(0.0f, 0.0f, 300.0f);
const FVector DemoUavMoveLocalOffsetCm(80.0f, -260.0f, 0.0f);
constexpr float DemoPedYawOffsetDeg = 180.0f;
constexpr float DemoVehicleYawOffsetDeg = 0.0f;
constexpr float DemoUavYawOffsetDeg = 30.0f;
constexpr float DemoUavMoveVelocityMps = 5.0f;

FVector ApplyRoadLocalOffset(const FTransform& RoadAnchorWorld, const FVector& LocalOffsetCm)
{
	return RoadAnchorWorld.TransformPositionNoScale(LocalOffsetCm);
}

float ResolveRoadRelativeYawDeg(const FTransform& RoadAnchorWorld, const float RelativeYawDeg)
{
	return FRotator::NormalizeAxis(RoadAnchorWorld.Rotator().Yaw + RelativeYawDeg);
}

TArray<TSharedPtr<FJsonValue>> MakeNumberArrayMeters(const FVector& EnuMeters)
{
	return {
		MakeShared<FJsonValueNumber>(EnuMeters.X),
		MakeShared<FJsonValueNumber>(EnuMeters.Y),
		MakeShared<FJsonValueNumber>(EnuMeters.Z)};
}

FString SerializeJsonObject(const TSharedPtr<FJsonObject>& Object)
{
	FString Output;
	const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	FJsonSerializer::Serialize(Object.ToSharedRef(), Writer);
	return Output;
}

bool ParseJsonString(const FString& JsonText, TSharedPtr<FJsonObject>& OutObject)
{
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
	return FJsonSerializer::Deserialize(Reader, OutObject) && OutObject.IsValid();
}

FString FormatStepState(const ECityValidationStepState State)
{
	switch (State)
	{
	case ECityValidationStepState::Running:
		return TEXT("RUNNING");
	case ECityValidationStepState::Unavailable:
		return TEXT("UNAVAILABLE");
	case ECityValidationStepState::Passed:
		return TEXT("PASSED");
	case ECityValidationStepState::Failed:
		return TEXT("FAILED");
	default:
		return TEXT("IDLE");
	}
}

}

UCityRuntimeValidationSubsystem* FindValidationSubsystem(UWorld* PreferredWorld = nullptr)
{
	if (PreferredWorld != nullptr)
	{
		return PreferredWorld->GetSubsystem<UCityRuntimeValidationSubsystem>();
	}

	if (GEngine == nullptr)
	{
		return nullptr;
	}

	for (const FWorldContext& WorldContext : GEngine->GetWorldContexts())
	{
		UWorld* World = WorldContext.World();
		if (World == nullptr || !(World->IsGameWorld() || World->WorldType == EWorldType::PIE))
		{
			continue;
		}

		if (UCityRuntimeValidationSubsystem* Subsystem = World->GetSubsystem<UCityRuntimeValidationSubsystem>())
		{
			return Subsystem;
		}
	}

	return nullptr;
}

class FCityRuntimeValidationInputProcessor : public IInputProcessor
{
public:
	explicit FCityRuntimeValidationInputProcessor(TWeakObjectPtr<UCityRuntimeValidationSubsystem> InSubsystem)
		: ValidationSubsystem(InSubsystem)
	{
	}

	virtual void Tick(const float, FSlateApplication&, TSharedRef<ICursor>) override
	{
	}

	virtual bool HandleKeyDownEvent(FSlateApplication&, const FKeyEvent& InKeyEvent) override
	{
		if (InKeyEvent.GetKey() != EKeys::F7 || InKeyEvent.IsRepeat())
		{
			return false;
		}

		if (UCityRuntimeValidationSubsystem* Subsystem = ValidationSubsystem.Get())
		{
			Subsystem->TogglePanelVisibility();
			return true;
		}

		return false;
	}

	virtual const TCHAR* GetDebugName() const override
	{
		return TEXT("CityRuntimeValidationInputProcessor");
	}

private:
	TWeakObjectPtr<UCityRuntimeValidationSubsystem> ValidationSubsystem;
};

static FAutoConsoleCommandWithWorldAndArgs GToggleValidationHudCmd(
	TEXT("aero.toggle_validation_hud"),
	TEXT("Toggle the PIE runtime validation HUD."),
	FConsoleCommandWithWorldAndArgsDelegate::CreateLambda(
		[](const TArray<FString>&, UWorld* World)
		{
			if (UCityRuntimeValidationSubsystem* Subsystem = FindValidationSubsystem(World))
			{
				Subsystem->TogglePanelVisibility();
			}
		}));

bool UCityRuntimeValidationSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UCityRuntimeValidationSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	ResetStepResults();
	RegisterInputPreProcessor();
}

void UCityRuntimeValidationSubsystem::Deinitialize()
{
	UnregisterInputPreProcessor();
	RemovePanel();
	PanelContainerWidget.Reset();
	PanelWidget.Reset();
	ResetDemoScheduling();
	TrackedObjects.Reset();
	RecentFeedbackEvents.Reset();
	Super::Deinitialize();
}

void UCityRuntimeValidationSubsystem::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	EnsurePanel();
	ExecuteScheduledActions();
	PollActiveHudUavMove();
	UpdateTrackedObjects();
}

TStatId UCityRuntimeValidationSubsystem::GetStatId() const
{
	RETURN_QUICK_DECLARE_CYCLE_STAT(UCityRuntimeValidationSubsystem, STATGROUP_Tickables);
}

void UCityRuntimeValidationSubsystem::EnsurePanel()
{
	if (!bPanelVisible || bPanelAttached || GEngine == nullptr || GEngine->GameViewport == nullptr)
	{
		return;
	}

	if (!PanelWidget.IsValid())
	{
		SAssignNew(PanelWidget, SCityRuntimeValidationPanel)
			.ValidationSubsystem(this);
	}

	if (!PanelContainerWidget.IsValid() && PanelWidget.IsValid())
	{
		PanelContainerWidget = SNew(SWeakWidget).PossiblyNullContent(PanelWidget.ToSharedRef());
	}

	if (PanelContainerWidget.IsValid())
	{
		GEngine->GameViewport->AddViewportWidgetContent(PanelContainerWidget.ToSharedRef(), 50);
		bPanelAttached = true;
	}
}

void UCityRuntimeValidationSubsystem::RemovePanel()
{
	if (bPanelAttached && GEngine != nullptr && GEngine->GameViewport != nullptr && PanelContainerWidget.IsValid())
	{
		GEngine->GameViewport->RemoveViewportWidgetContent(PanelContainerWidget.ToSharedRef());
	}

	bPanelAttached = false;
}

void UCityRuntimeValidationSubsystem::TogglePanelVisibility()
{
	bPanelVisible = !bPanelVisible;
	if (bPanelVisible)
	{
		EnsurePanel();
	}
	else
	{
		RemovePanel();
	}
}

void UCityRuntimeValidationSubsystem::RegisterInputPreProcessor()
{
	if (InputProcessor.IsValid() || !FSlateApplication::IsInitialized())
	{
		return;
	}

	InputProcessor = MakeShared<FCityRuntimeValidationInputProcessor>(this);
	FSlateApplication::Get().RegisterInputPreProcessor(InputProcessor);
}

void UCityRuntimeValidationSubsystem::UnregisterInputPreProcessor()
{
	if (!InputProcessor.IsValid())
	{
		return;
	}

	if (FSlateApplication::IsInitialized())
	{
		FSlateApplication::Get().UnregisterInputPreProcessor(InputProcessor);
	}

	InputProcessor.Reset();
}

void UCityRuntimeValidationSubsystem::ResetValidationState()
{
	ResetStepResults();
	RecentFeedbackEvents.Reset();
	LastErrorMessage.Reset();
	OverallState = ECityValidationStepState::Idle;
	bDemoRunning = false;
	ActiveHudVehicleName.Reset();
	ActiveHudUavName.Reset();
}

void UCityRuntimeValidationSubsystem::ResetStepResults()
{
	StepResults.Reset();
	const TArray<FString> StepNames = {
		TEXT("Load Context"),
		TEXT("Spawn Ped"),
		TEXT("Set Variant"),
		TEXT("Observe"),
		TEXT("Commit Cross"),
		TEXT("Spawn Crowd"),
		TEXT("Spawn Cone"),
		TEXT("Spawn StreetLight Placeholder"),
		StepSpawnVehicle,
		TEXT("Spawn Runtime UAV"),
		StepMoveVehicle,
		TEXT("Move UAV"),
		TEXT("Poll Feedback"),
		TEXT("Recheck Grounding")};
	for (const FString& StepName : StepNames)
	{
		FCityValidationStepResult& Result = StepResults.AddDefaulted_GetRef();
		Result.Name = StepName;
	}
}

void UCityRuntimeValidationSubsystem::ResetDemoScheduling()
{
	ScheduledActions.Reset();
}

void UCityRuntimeValidationSubsystem::ScheduleAction(double DelaySeconds, TFunction<void()> Action)
{
	FCityScheduledAction& ScheduledAction = ScheduledActions.AddDefaulted_GetRef();
	ScheduledAction.ExecuteAt = FPlatformTime::Seconds() + DelaySeconds;
	ScheduledAction.Action = MoveTemp(Action);
}

void UCityRuntimeValidationSubsystem::ExecuteScheduledActions()
{
	const double Now = FPlatformTime::Seconds();
	for (int32 Index = ScheduledActions.Num() - 1; Index >= 0; --Index)
	{
		if (ScheduledActions[Index].ExecuteAt <= Now)
		{
			TFunction<void()> Action = MoveTemp(ScheduledActions[Index].Action);
			ScheduledActions.RemoveAt(Index);
			if (Action)
			{
				Action();
			}
		}
	}
}

void UCityRuntimeValidationSubsystem::PollActiveHudUavMove()
{
	bool bMoveStepRunning = false;
	for (const FCityValidationStepResult& StepResult : StepResults)
	{
		if (StepResult.Name == TEXT("Move UAV") && StepResult.State == ECityValidationStepState::Running)
		{
			bMoveStepRunning = true;
			break;
		}
	}

	if (!bMoveStepRunning || ActiveHudUavName.IsEmpty())
	{
		return;
	}

	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FAeroRuntimeMoveStatus MoveStatus;
	FString Error;
	if (!RuntimeSubsystem->GetMultirotorMoveStatus(ActiveHudUavName, MoveStatus, Error))
	{
		LastErrorMessage = Error;
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	switch (MoveStatus.State)
	{
	case EAeroRuntimeMoveState::Running:
	case EAeroRuntimeMoveState::Idle:
		return;
	case EAeroRuntimeMoveState::Succeeded:
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Passed, MoveStatus.Message.IsEmpty() ? TEXT("AirSim UAV move completed.") : MoveStatus.Message);
		if (!bDemoRunning && OverallState != ECityValidationStepState::Failed)
		{
			bool bHasRunningStep = false;
			for (const FCityValidationStepResult& StepResult : StepResults)
			{
				if (StepResult.State == ECityValidationStepState::Running)
				{
					bHasRunningStep = true;
					break;
				}
			}
			if (!bHasRunningStep)
			{
				SetOverallState(ECityValidationStepState::Passed, TEXT("Full demo finished."));
			}
		}
		return;
	case EAeroRuntimeMoveState::Cancelled:
		LastErrorMessage = MoveStatus.Message.IsEmpty() ? TEXT("AirSim UAV move was cancelled.") : MoveStatus.Message;
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	case EAeroRuntimeMoveState::Failed:
	default:
		LastErrorMessage = MoveStatus.Message.IsEmpty() ? TEXT("AirSim UAV move failed.") : MoveStatus.Message;
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}
}

void UCityRuntimeValidationSubsystem::SetOverallState(ECityValidationStepState NewState, const FString& NewErrorMessage)
{
	OverallState = NewState;
	if (!NewErrorMessage.IsEmpty())
	{
		LastErrorMessage = NewErrorMessage;
	}
}

void UCityRuntimeValidationSubsystem::SetStepState(const FString& StepName, ECityValidationStepState NewState, const FString& Message)
{
	for (FCityValidationStepResult& StepResult : StepResults)
	{
		if (StepResult.Name == StepName)
		{
			StepResult.State = NewState;
			StepResult.Message = Message;
			UE_LOG(
				LogTemp,
				Log,
				TEXT("CityRuntimeValidation step: name='%s' state='%s' message='%s'."),
				*StepName,
				*FormatStepState(NewState),
				Message.IsEmpty() ? TEXT("<none>") : *Message);
			if (NewState == ECityValidationStepState::Failed)
			{
				SetOverallState(ECityValidationStepState::Failed, Message);
			}
			else if (NewState == ECityValidationStepState::Running)
			{
				SetOverallState(ECityValidationStepState::Running, Message);
			}
			return;
		}
	}
}

void UCityRuntimeValidationSubsystem::SetStepUnavailable(const FString& StepName, const FString& Message)
{
	SetStepState(StepName, ECityValidationStepState::Unavailable, Message);
}

FString UCityRuntimeValidationSubsystem::GetHeaderText() const
{
	return FString::Printf(TEXT("PIE Runtime Validation HUD [%s]"), *FormatStepState(OverallState));
}

FSlateColor UCityRuntimeValidationSubsystem::GetHeaderColor() const
{
	switch (OverallState)
	{
	case ECityValidationStepState::Passed:
		return FSlateColor(FLinearColor(0.35f, 0.95f, 0.45f));
	case ECityValidationStepState::Failed:
		return FSlateColor(FLinearColor(0.95f, 0.35f, 0.35f));
	case ECityValidationStepState::Running:
		return FSlateColor(FLinearColor(0.95f, 0.85f, 0.25f));
	case ECityValidationStepState::Unavailable:
		return FSlateColor(FLinearColor(0.65f, 0.75f, 0.95f));
	default:
		return FSlateColor(FLinearColor::White);
	}
}

FString UCityRuntimeValidationSubsystem::GetStatusSummaryText() const
{
	return FString::Printf(
		TEXT("Map: %s\nState: %s\nError: %s\nHotkey: F7 | Console: aero.toggle_validation_hud"),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*FormatStepState(OverallState),
		LastErrorMessage.IsEmpty() ? TEXT("<none>") : *LastErrorMessage);
}

FString UCityRuntimeValidationSubsystem::GetCapabilitiesText() const
{
	return TEXT("Ped Variants: adult_male_commuter, adult_female_commuter, child_crossing, elder_observer\nModes: idle, stop, observe, start_cross, cross\nMontages: observe, start_cross, stop");
}

FString UCityRuntimeValidationSubsystem::GetStepResultsText() const
{
	FString Output;
	for (const FCityValidationStepResult& StepResult : StepResults)
	{
		Output += FString::Printf(TEXT("[%s] %s"), *FormatStepState(StepResult.State), *StepResult.Name);
		if (!StepResult.Message.IsEmpty())
		{
			Output += FString::Printf(TEXT(" - %s"), *StepResult.Message);
		}
		Output += LINE_TERMINATOR;
	}
	return Output.IsEmpty() ? TEXT("<no steps>") : Output;
}

FString UCityRuntimeValidationSubsystem::GetTrackedCountsText() const
{
	int32 PedCount = 0;
	int32 CrowdCount = 0;
	int32 AssetCount = 0;
	int32 RuntimeVehicleCount = 0;
	int32 CollisionCount = 0;

	for (const FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		switch (TrackedObject.Kind)
		{
		case ECityTrackedObjectKind::Pedestrian:
			++PedCount;
			break;
		case ECityTrackedObjectKind::Crowd:
			++CrowdCount;
			break;
		case ECityTrackedObjectKind::Asset:
			++AssetCount;
			break;
		case ECityTrackedObjectKind::RuntimeVehicle:
			++RuntimeVehicleCount;
			break;
		}
	}

	for (const FAeroFeedbackEvent& Event : RecentFeedbackEvents)
	{
		if (Event.Type.Equals(TEXT("collision"), ESearchCase::IgnoreCase))
		{
			++CollisionCount;
		}
	}

	return FString::Printf(TEXT("ped=%d | crowd=%d | assets=%d | runtime=%d | collisions=%d"), PedCount, CrowdCount, AssetCount, RuntimeVehicleCount, CollisionCount);
}

FString UCityRuntimeValidationSubsystem::GetFeedbackText() const
{
	if (RecentFeedbackEvents.Num() == 0)
	{
		return TEXT("<no feedback events>");
	}

	FString Output;
	for (const FAeroFeedbackEvent& Event : RecentFeedbackEvents)
	{
		Output += BuildFeedbackSummary(Event) + LINE_TERMINATOR;
	}
	return Output;
}

FString UCityRuntimeValidationSubsystem::GetPassedObjectsText() const
{
	FString Output;
	for (const FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (TrackedObject.Actor.IsValid() && TrackedObject.bGrounded && TrackedObject.bHasFeedback)
		{
			Output += FString::Printf(TEXT("%s [%s] %s"), *TrackedObject.Id, *TrackedObject.LogicalAssetId, *TrackedObject.LastFeedbackType) + LINE_TERMINATOR;
		}
	}
	return Output.IsEmpty() ? TEXT("<none>") : Output;
}

FString UCityRuntimeValidationSubsystem::GetPendingObjectsText() const
{
	FString Output;
	for (const FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (TrackedObject.Actor.IsValid() && TrackedObject.bGrounded && !TrackedObject.bHasFeedback)
		{
			Output += FString::Printf(TEXT("%s [%s] grounded, waiting feedback"), *TrackedObject.Id, *TrackedObject.LogicalAssetId) + LINE_TERMINATOR;
		}
	}
	return Output.IsEmpty() ? TEXT("<none>") : Output;
}

FString UCityRuntimeValidationSubsystem::GetFailedObjectsText() const
{
	FString Output;
	for (const FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (!TrackedObject.Actor.IsValid())
		{
			Output += FString::Printf(TEXT("%s [%s] actor missing"), *TrackedObject.Id, *TrackedObject.LogicalAssetId) + LINE_TERMINATOR;
		}
		else if (!TrackedObject.bGrounded)
		{
			Output += FString::Printf(TEXT("%s [%s] %s"), *TrackedObject.Id, *TrackedObject.LogicalAssetId, *TrackedObject.GroundingMessage) + LINE_TERMINATOR;
		}
	}
	return Output.IsEmpty() ? TEXT("<none>") : Output;
}

bool UCityRuntimeValidationSubsystem::ResolveDemoRoadAnchor(FTransform& OutRoadAnchorWorld, FSumoNearestLaneSample* OutRoadSample)
{
	OutRoadAnchorWorld = FTransform::Identity;
	if (OutRoadSample != nullptr)
	{
		*OutRoadSample = FSumoNearestLaneSample();
	}

	if (GetWorld() == nullptr)
	{
		LastErrorMessage = TEXT("Cannot resolve demo road anchor: World is null.");
		return false;
	}

	FVector QueryWorldCm = GetCurrentWorldOriginCm();
	const APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
	const AActor* QueryActor = PlayerController != nullptr ? PlayerController->GetViewTarget() : nullptr;
	if (!IsValid(QueryActor))
	{
		QueryActor = PlayerController != nullptr ? PlayerController->GetPawn() : nullptr;
	}
	if (IsValid(QueryActor))
	{
		QueryWorldCm = QueryActor->GetActorLocation();
	}

	FSumoNearestLaneSample RoadSample;
	FString Error;
	if (!FSumoRoadTopologyQuery::FindNearestRoadSample(GetWorld(), QueryWorldCm, RoadSample, Error))
	{
		LastErrorMessage = Error.IsEmpty()
			? TEXT("Failed to resolve demo road anchor from SUMO topology.")
			: Error;
		return false;
	}

	OutRoadAnchorWorld = RoadSample.WorldTransform;
	if (OutRoadSample != nullptr)
	{
		*OutRoadSample = RoadSample;
	}

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation resolved road anchor: query='%s' lane='%s' edge='%s' dist_2d_cm=%.2f anchor='%s' yaw=%.2f."),
		*QueryWorldCm.ToString(),
		*RoadSample.LaneId,
		*RoadSample.EdgeId,
		RoadSample.Distance2DCm,
		*RoadSample.WorldTransform.GetLocation().ToString(),
		RoadSample.WorldTransform.Rotator().Yaw);
	return true;
}

FVector UCityRuntimeValidationSubsystem::GetDemoOriginWorldCm() const
{
	APlayerController* PlayerController = GetWorld() != nullptr ? GetWorld()->GetFirstPlayerController() : nullptr;
	APawn* Pawn = PlayerController != nullptr ? PlayerController->GetPawn() : nullptr;
	if (Pawn != nullptr)
	{
		FVector DemoOriginWorldCm = Pawn->GetActorLocation();
		if (TryFindNearestPreferredGroundWorldCm(DemoOriginWorldCm, DemoGroundSearchRadiusCm, DemoGroundSearchStepCm, DemoOriginWorldCm))
		{
			return DemoOriginWorldCm;
		}
		return SnapPointToGround(Pawn->GetActorLocation());
	}
	return SnapPointToGround(FVector::ZeroVector);
}

bool UCityRuntimeValidationSubsystem::GetDemoReferenceWorldCm(FVector& OutWorldCm) const
{
	OutWorldCm = FVector::ZeroVector;
	if (GetWorld() == nullptr)
	{
		return false;
	}

	const APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
	const APawn* Pawn = PlayerController != nullptr ? PlayerController->GetPawn() : nullptr;
	if (Pawn == nullptr)
	{
		return false;
	}

	const FVector PawnWorldCm = Pawn->GetActorLocation();
	if (TryFindNearestPreferredGroundWorldCm(PawnWorldCm, DemoGroundSearchRadiusCm, DemoGroundSearchStepCm, OutWorldCm))
	{
		return true;
	}

	return AeroGroundPlacement::TryProjectWorldPointToGround(GetWorld(), PawnWorldCm, OutWorldCm, nullptr, Pawn);
}

FVector UCityRuntimeValidationSubsystem::SnapPointToGround(const FVector& WorldLocationCm) const
{
	AeroGroundPlacement::FResolvedGroundPlacement Placement;
	if (AeroGroundPlacement::ResolveGroundPlacement(GetWorld(), WorldLocationCm, Placement))
	{
		return Placement.GroundWorldCm;
	}

	FVector SnappedLocation = WorldLocationCm;
	TryFindNearestPreferredGroundWorldCm(WorldLocationCm, DemoGroundSearchRadiusCm, DemoGroundSearchStepCm, SnappedLocation);
	return SnappedLocation;
}

bool UCityRuntimeValidationSubsystem::TryFindNearestPreferredGroundWorldCm(const FVector& OriginWorldCm, float SearchRadiusCm, float StepCm, FVector& OutWorldCm) const
{
	if (GetWorld() == nullptr)
	{
		return false;
	}

	bool bFoundGround = false;
	double BestDistanceSquared = TNumericLimits<double>::Max();
	const float ClampedStepCm = FMath::Max(1.0f, StepCm);
	const int32 Steps = FMath::Max(1, FMath::RoundToInt(SearchRadiusCm / ClampedStepCm));

	auto TryCandidate = [this, &OriginWorldCm, &OutWorldCm, &bFoundGround, &BestDistanceSquared](const FVector& CandidateWorldCm)
	{
		FVector ProjectedWorldCm = CandidateWorldCm;
		if (!AeroGroundPlacement::TryProjectWorldPointToGround(GetWorld(), CandidateWorldCm, ProjectedWorldCm))
		{
			return;
		}

		const double DistanceSquared = FVector::DistSquared2D(ProjectedWorldCm, OriginWorldCm);
		if (!bFoundGround || DistanceSquared < BestDistanceSquared)
		{
			bFoundGround = true;
			BestDistanceSquared = DistanceSquared;
			OutWorldCm = ProjectedWorldCm;
		}
	};

	TryCandidate(OriginWorldCm);
	for (int32 XIndex = -Steps; XIndex <= Steps; ++XIndex)
	{
		for (int32 YIndex = -Steps; YIndex <= Steps; ++YIndex)
		{
			if (XIndex == 0 && YIndex == 0)
			{
				continue;
			}

			const FVector Candidate = OriginWorldCm + FVector(XIndex * ClampedStepCm, YIndex * ClampedStepCm, 0.0f);
			TryCandidate(Candidate);
		}
	}

	return bFoundGround;
}

FVector UCityRuntimeValidationSubsystem::FindNearestPreferredGroundWorldCm(const FVector& OriginWorldCm, float SearchRadiusCm, float StepCm) const
{
	FVector BestLocation = SnapPointToGround(OriginWorldCm);
	TryFindNearestPreferredGroundWorldCm(OriginWorldCm, SearchRadiusCm, StepCm, BestLocation);
	return BestLocation;
}

bool UCityRuntimeValidationSubsystem::QueryNearestTaggedInstanceWorldCm(const FString& QueryTag, const FVector& NearWorldCm, FVector& OutWorldCm) const
{
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("tag"), QueryTag);
	Payload->SetArrayField(TEXT("pose_enu_m"), MakeNumberArrayMeters(WorldCmToEnuM(NearWorldCm)));
	Payload->SetNumberField(TEXT("radius_m"), 200.0);

	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("QueryNearest"), Payload, ResponsePayload, Error))
	{
		return false;
	}

	if (!ResponsePayload.IsValid() || !ResponsePayload->HasField(TEXT("found")) || !ResponsePayload->GetBoolField(TEXT("found")))
	{
		return false;
	}

	if (!ResponsePayload->HasTypedField<EJson::Array>(TEXT("position_enu_m")))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>& Values = ResponsePayload->GetArrayField(TEXT("position_enu_m"));
	if (Values.Num() < 3)
	{
		return false;
	}

	const FVector PositionEnuM = FVector(Values[0]->AsNumber(), Values[1]->AsNumber(), Values[2]->AsNumber());
	OutWorldCm = GetCurrentWorldOriginCm() + PositionEnuM * 100.0f;
	return true;
}

FVector UCityRuntimeValidationSubsystem::WorldCmToEnuM(const FVector& WorldLocationCm) const
{
	return (WorldLocationCm - GetCurrentWorldOriginCm()) / 100.0f;
}

bool UCityRuntimeValidationSubsystem::CallBridge(const FString& OperationName, const TSharedPtr<FJsonObject>& Payload, TSharedPtr<FJsonObject>& OutPayload, FString& OutError) const
{
	UAeroBridgeWorldSubsystem* BridgeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroBridgeWorldSubsystem>() : nullptr;
	if (BridgeSubsystem == nullptr)
	{
		OutError = TEXT("AeroBridgeWorldSubsystem unavailable.");
		return false;
	}

	TSharedPtr<FJsonObject> RootObject = MakeShared<FJsonObject>();
	RootObject->SetStringField(TEXT("api_version"), TEXT("1.0"));
	RootObject->SetStringField(TEXT("request_id"), TEXT("hud"));
	if (!CurrentMapId.IsEmpty())
	{
		RootObject->SetStringField(TEXT("map_id"), CurrentMapId);
	}
	RootObject->SetObjectField(TEXT("payload"), Payload.IsValid() ? Payload : MakeShared<FJsonObject>());

	const FString RequestJson = SerializeJsonObject(RootObject);
	FString ResponseJson;
	if (OperationName == TEXT("LoadContext"))
	{
		ResponseJson = BridgeSubsystem->HandleLoadContext(RequestJson);
	}
	else if (OperationName == TEXT("SpawnAsset"))
	{
		ResponseJson = BridgeSubsystem->HandleSpawnAsset(RequestJson);
	}
	else if (OperationName == TEXT("MoveAsset"))
	{
		ResponseJson = BridgeSubsystem->HandleMoveAsset(RequestJson);
	}
	else if (OperationName == TEXT("RemoveAsset"))
	{
		ResponseJson = BridgeSubsystem->HandleRemoveAsset(RequestJson);
	}
	else if (OperationName == TEXT("QueryNearest"))
	{
		ResponseJson = BridgeSubsystem->HandleQueryNearest(RequestJson);
	}
	else if (OperationName == TEXT("ApplyFrame"))
	{
		ResponseJson = BridgeSubsystem->HandleApplyFrame(RequestJson);
	}
	else
	{
		OutError = FString::Printf(TEXT("Unsupported bridge operation: %s"), *OperationName);
		return false;
	}

	TSharedPtr<FJsonObject> ResponseObject;
	if (!ParseJsonString(ResponseJson, ResponseObject))
	{
		OutError = TEXT("Bridge returned invalid JSON.");
		return false;
	}

	const FString Status = ResponseObject->GetStringField(TEXT("status"));
	if (!Status.Equals(TEXT("ok"), ESearchCase::IgnoreCase))
	{
		if (ResponseObject->HasTypedField<EJson::Object>(TEXT("error")))
		{
			OutError = ResponseObject->GetObjectField(TEXT("error"))->GetStringField(TEXT("message"));
		}
		else
		{
			OutError = ResponseJson;
		}
		return false;
	}

	OutPayload = ResponseObject->HasTypedField<EJson::Object>(TEXT("payload")) ? ResponseObject->GetObjectField(TEXT("payload")) : MakeShared<FJsonObject>();
	return true;
}

bool UCityRuntimeValidationSubsystem::LoadContextInternal(const FString& MapId)
{
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("map_id"), MapId);
	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("LoadContext"), Payload, ResponsePayload, Error))
	{
		LastErrorMessage = Error;
		return false;
	}

	CurrentMapId = MapId;
	return true;
}

bool UCityRuntimeValidationSubsystem::ApplySceneFrameInternal(const TSharedPtr<FJsonObject>& FramePayload)
{
	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("ApplyFrame"), FramePayload, ResponsePayload, Error))
	{
		LastErrorMessage = Error;
		return false;
	}

	return true;
}

bool UCityRuntimeValidationSubsystem::SpawnSceneVehicleInternal(
	const FString& EntityId,
	const FString& ProxyTemplateId,
	const FVector& WorldLocationCm,
	const float YawDeg)
{
	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	const FVector PositionEnuM = WorldCmToEnuM(WorldLocationCm);
	TSharedPtr<FJsonObject> SpawnDelta = MakeShared<FJsonObject>();
	SpawnDelta->SetStringField(TEXT("entity_id"), EntityId);
	SpawnDelta->SetStringField(TEXT("proxy_template_id"), ProxyTemplateId);

	TSharedPtr<FJsonObject> PoseObject = MakeShared<FJsonObject>();
	PoseObject->SetArrayField(TEXT("position_enu_m"), MakeNumberArrayMeters(PositionEnuM));
	TSharedPtr<FJsonObject> RotationObject = MakeShared<FJsonObject>();
	RotationObject->SetNumberField(TEXT("roll_deg"), 0.0);
	RotationObject->SetNumberField(TEXT("pitch_deg"), 0.0);
	RotationObject->SetNumberField(TEXT("yaw_deg"), YawDeg);
	PoseObject->SetObjectField(TEXT("rotation_deg"), RotationObject);
	SpawnDelta->SetObjectField(TEXT("pose_enu_m"), PoseObject);
	TArray<TSharedPtr<FJsonValue>> TagValues;
	TagValues.Add(MakeShared<FJsonValueString>(TEXT("vehicle")));
	TagValues.Add(MakeShared<FJsonValueString>(TEXT("hud")));
	TagValues.Add(MakeShared<FJsonValueString>(TEXT("scene_sync")));
	SpawnDelta->SetArrayField(TEXT("tags"), TagValues);

	TSharedPtr<FJsonObject> FramePayload = MakeShared<FJsonObject>();
	TArray<TSharedPtr<FJsonValue>> SpawnValues;
	SpawnValues.Add(MakeShared<FJsonValueObject>(SpawnDelta));
	FramePayload->SetArrayField(TEXT("spawns"), SpawnValues);

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnSceneVehicleInternal request: map_id='%s' entity_id='%s' proxy_template_id='%s' world='%s' enu_m='%s' yaw=%.2f."),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*EntityId,
		*ProxyTemplateId,
		*WorldLocationCm.ToString(),
		*PositionEnuM.ToString(),
		YawDeg);

	return ApplySceneFrameInternal(FramePayload);
}

bool UCityRuntimeValidationSubsystem::MoveSceneVehicleInternal(const FString& EntityId, const FVector& WorldLocationCm, const float YawDeg)
{
	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	const FVector PositionEnuM = WorldCmToEnuM(WorldLocationCm);
	TSharedPtr<FJsonObject> UpdateDelta = MakeShared<FJsonObject>();
	UpdateDelta->SetStringField(TEXT("entity_id"), EntityId);

	TSharedPtr<FJsonObject> PoseObject = MakeShared<FJsonObject>();
	PoseObject->SetArrayField(TEXT("position_enu_m"), MakeNumberArrayMeters(PositionEnuM));
	TSharedPtr<FJsonObject> RotationObject = MakeShared<FJsonObject>();
	RotationObject->SetNumberField(TEXT("roll_deg"), 0.0);
	RotationObject->SetNumberField(TEXT("pitch_deg"), 0.0);
	RotationObject->SetNumberField(TEXT("yaw_deg"), YawDeg);
	PoseObject->SetObjectField(TEXT("rotation_deg"), RotationObject);
	UpdateDelta->SetObjectField(TEXT("pose_enu_m"), PoseObject);

	TSharedPtr<FJsonObject> FramePayload = MakeShared<FJsonObject>();
	TArray<TSharedPtr<FJsonValue>> UpdateValues;
	UpdateValues.Add(MakeShared<FJsonValueObject>(UpdateDelta));
	FramePayload->SetArrayField(TEXT("updates"), UpdateValues);

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation MoveSceneVehicleInternal request: map_id='%s' entity_id='%s' world='%s' enu_m='%s' yaw=%.2f."),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*EntityId,
		*WorldLocationCm.ToString(),
		*PositionEnuM.ToString(),
		YawDeg);

	return ApplySceneFrameInternal(FramePayload);
}

bool UCityRuntimeValidationSubsystem::RemoveSceneEntityInternal(const FString& EntityId)
{
	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	TSharedPtr<FJsonObject> RemoveDelta = MakeShared<FJsonObject>();
	RemoveDelta->SetStringField(TEXT("entity_id"), EntityId);

	TSharedPtr<FJsonObject> FramePayload = MakeShared<FJsonObject>();
	TArray<TSharedPtr<FJsonValue>> RemoveValues;
	RemoveValues.Add(MakeShared<FJsonValueObject>(RemoveDelta));
	FramePayload->SetArrayField(TEXT("removes"), RemoveValues);

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation RemoveSceneEntityInternal request: map_id='%s' entity_id='%s'."),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*EntityId);

	return ApplySceneFrameInternal(FramePayload);
}

FVector UCityRuntimeValidationSubsystem::GetCurrentWorldOriginCm() const
{
	const UAeroBridgeWorldSubsystem* BridgeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroBridgeWorldSubsystem>() : nullptr;
	const TSharedPtr<FJsonObject> MapContext = BridgeSubsystem != nullptr ? BridgeSubsystem->GetCurrentMapContext() : nullptr;
	FVector WorldOriginCm = FVector::ZeroVector;
	if (MapContext.IsValid() && MapContext->HasTypedField<EJson::Array>(TEXT("world_origin_cm")))
	{
		const TArray<TSharedPtr<FJsonValue>>& Values = MapContext->GetArrayField(TEXT("world_origin_cm"));
		if (Values.Num() >= 3)
		{
			WorldOriginCm = FVector(Values[0]->AsNumber(), Values[1]->AsNumber(), Values[2]->AsNumber());
		}
	}
	return WorldOriginCm;
}

bool UCityRuntimeValidationSubsystem::SpawnAssetInternal(
	const FString& AssetId,
	const FString& LogicalAssetId,
	const FVector& WorldLocationCm,
	float YawDeg,
	const bool bSnapToGround,
	FString* OutResolvedAssetId)
{
	(void)bSnapToGround;

	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	const FVector ResolvedWorldLocationCm = WorldLocationCm;
	const FVector ResolvedEnuMeters = WorldCmToEnuM(ResolvedWorldLocationCm);
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("map_id"), CurrentMapId);
	Payload->SetStringField(TEXT("asset_id"), AssetId);
	Payload->SetStringField(TEXT("logical_asset_id"), LogicalAssetId);
	Payload->SetArrayField(TEXT("position_enu_m"), MakeNumberArrayMeters(ResolvedEnuMeters));
	Payload->SetNumberField(TEXT("yaw_deg"), YawDeg);

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnAssetInternal request: map_id='%s' asset_id='%s' logical_asset_id='%s' requested_world='%s' payload_world='%s' payload_enu_m='%s' yaw=%.2f local_snap_skipped=%s."),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*AssetId,
		*LogicalAssetId,
		*WorldLocationCm.ToString(),
		*ResolvedWorldLocationCm.ToString(),
		*ResolvedEnuMeters.ToString(),
		YawDeg,
		TEXT("true"));

	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("SpawnAsset"), Payload, ResponsePayload, Error))
	{
		UE_LOG(
			LogTemp,
			Warning,
			TEXT("CityRuntimeValidation SpawnAssetInternal failed: asset_id='%s' logical_asset_id='%s' error='%s'."),
			*AssetId,
			*LogicalAssetId,
			*Error);
		LastErrorMessage = Error;
		return false;
	}

	FString ResolvedAssetId = AssetId;
	ResponsePayload->TryGetStringField(TEXT("asset_id"), ResolvedAssetId);
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnAssetInternal success: requested_asset_id='%s' resolved_asset_id='%s' logical_asset_id='%s'."),
		*AssetId,
		*ResolvedAssetId,
		*LogicalAssetId);

	if (OutResolvedAssetId != nullptr)
	{
		*OutResolvedAssetId = ResolvedAssetId;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::MoveAssetInternal(const FString& AssetId, const FVector& WorldLocationCm, float YawDeg, const bool bSnapToGround)
{
	(void)bSnapToGround;

	if (CurrentMapId.IsEmpty() && !LoadContextInternal(DemoMapId))
	{
		return false;
	}

	const FVector ResolvedWorldLocationCm = WorldLocationCm;
	const FVector ResolvedEnuMeters = WorldCmToEnuM(ResolvedWorldLocationCm);
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("map_id"), CurrentMapId);
	Payload->SetStringField(TEXT("asset_id"), AssetId);
	Payload->SetArrayField(TEXT("position_enu_m"), MakeNumberArrayMeters(ResolvedEnuMeters));
	Payload->SetNumberField(TEXT("yaw_deg"), YawDeg);

	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation MoveAssetInternal request: map_id='%s' asset_id='%s' requested_world='%s' payload_world='%s' payload_enu_m='%s' yaw=%.2f local_snap_skipped=%s."),
		CurrentMapId.IsEmpty() ? TEXT("<none>") : *CurrentMapId,
		*AssetId,
		*WorldLocationCm.ToString(),
		*ResolvedWorldLocationCm.ToString(),
		*ResolvedEnuMeters.ToString(),
		YawDeg,
		TEXT("true"));

	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("MoveAsset"), Payload, ResponsePayload, Error))
	{
		UE_LOG(
			LogTemp,
			Warning,
			TEXT("CityRuntimeValidation MoveAssetInternal failed: asset_id='%s' error='%s'."),
			*AssetId,
			*Error);
		LastErrorMessage = Error;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("CityRuntimeValidation MoveAssetInternal success: asset_id='%s'."), *AssetId);
	return true;
}

bool UCityRuntimeValidationSubsystem::RemoveAssetInternal(const FString& AssetId)
{
	TSharedPtr<FJsonObject> Payload = MakeShared<FJsonObject>();
	Payload->SetStringField(TEXT("asset_id"), AssetId);

	TSharedPtr<FJsonObject> ResponsePayload;
	FString Error;
	if (!CallBridge(TEXT("RemoveAsset"), Payload, ResponsePayload, Error))
	{
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::SpawnPedInternal(
	const FString& PedId,
	const FVector& WorldLocationCm,
	float YawDeg,
	const FName& VariantId,
	const bool bUseProvidedGroundPoint)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	const FVector ResolvedWorldLocationCm = WorldLocationCm;
	if (!RuntimeSubsystem->SpawnPedestrian(PedId, ResolvedWorldLocationCm, YawDeg, VariantId, LastErrorMessage, bUseProvidedGroundPoint))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to spawn ped '%s'."), *PedId);
		}
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::SetPedVariantInternal(const FString& PedId, const FName& VariantId)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	if (!RuntimeSubsystem->SetPedestrianVariant(PedId, VariantId, LastErrorMessage))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to set variant '%s' on ped '%s'."), *VariantId.ToString(), *PedId);
		}
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::ObservePedInternal(const FString& PedId)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	if (!RuntimeSubsystem->ObservePedestrian(PedId, NAME_None, LastErrorMessage))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to observe ped '%s'."), *PedId);
		}
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::PlayPedAnimationInternal(const FString& PedId, const FString& AnimationAssetPath)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	if (!RuntimeSubsystem->PlayPedestrianAnimation(PedId, AnimationAssetPath, NAME_None, 1.0f, 1, LastErrorMessage))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to play animation '%s' on ped '%s'."), *AnimationAssetPath, *PedId);
		}
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::CommitCrossInternal(
	const FString& PedId,
	const FVector& TargetWorldCm,
	float SpeedCmPerSec,
	const bool bSnapToGround)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	const FVector ResolvedTargetWorldCm = bSnapToGround ? SnapPointToGround(TargetWorldCm) : TargetWorldCm;
	if (!RuntimeSubsystem->CommitPedestrianCross(PedId, ResolvedTargetWorldCm, SpeedCmPerSec, LastErrorMessage))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to commit cross for ped '%s'."), *PedId);
		}
		return false;
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::ReleasePedInternal(const FString& PedId)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	if (!RuntimeSubsystem->ReleasePedestrian(PedId, LastErrorMessage))
	{
		if (LastErrorMessage.IsEmpty())
		{
			LastErrorMessage = FString::Printf(TEXT("Failed to release ped '%s'."), *PedId);
		}
		return false;
	}

	return true;
}

bool UCityRuntimeValidationSubsystem::SpawnCrowdInternal(
	const FString& GroupId,
	int32 Count,
	const FVector& WorldOriginCm,
	const bool bUseProvidedGroundPoint)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		return false;
	}

	FCrowdSpawnRequest SpawnRequest;
	SpawnRequest.GroupId = FName(*GroupId);
	SpawnRequest.Count = Count;
	SpawnRequest.Seed = 77;
	SpawnRequest.SpawnOrigin = WorldOriginCm;
	SpawnRequest.SpawnBoxExtent = FVector(120.0f, 120.0f, 0.0f);
	SpawnRequest.YawPolicy = ECrowdYawPolicy::Random;
	SpawnRequest.bUseProvidedGroundPoint = bUseProvidedGroundPoint;
	FCrowdSpawnResult Result;
	if (!RuntimeSubsystem->SpawnCrowd(SpawnRequest, Result, LastErrorMessage))
	{
		return false;
	}
	if (Result.SpawnedIds.Num() == 0)
	{
		LastErrorMessage = TEXT("Crowd spawn returned no pedestrians.");
		return false;
	}

	for (const FString& SpawnedId : Result.SpawnedIds)
	{
		TrackPedestrian(SpawnedId, ECityTrackedObjectKind::Crowd, GroupId);
	}
	return true;
}

bool UCityRuntimeValidationSubsystem::ClearCrowdInternal(const FString& GroupId)
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		UntrackGroup(GroupId);
		return false;
	}

	const bool bCleared = RuntimeSubsystem != nullptr && RuntimeSubsystem->ClearCrowdGroup(FName(*GroupId), LastErrorMessage);
	UntrackGroup(GroupId);
	return bCleared;
}

AActor* UCityRuntimeValidationSubsystem::ResolvePedestrianActor(const FString& PedId) const
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	return RuntimeSubsystem != nullptr ? RuntimeSubsystem->ResolvePedestrianActor(PedId) : nullptr;
}

AActor* UCityRuntimeValidationSubsystem::ResolveAssetActor(const FString& AssetId) const
{
	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>() : nullptr;
	if (AssetSubsystem == nullptr)
	{
		return nullptr;
	}

	const FAeroAssetInstanceState* Instance = AssetSubsystem->FindInstance(AssetId);
	if (Instance != nullptr && Instance->Actor.IsValid())
	{
		return Instance->Actor.Get();
	}

	const FString ProxyId = FString::Printf(TEXT("entity_proxy_%s"), *AssetId);
	Instance = AssetSubsystem->FindInstance(ProxyId);
	return Instance != nullptr && Instance->Actor.IsValid() ? Instance->Actor.Get() : nullptr;
}

AActor* UCityRuntimeValidationSubsystem::ResolveRuntimeVehicleActor(const FString& VehicleName) const
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	return RuntimeSubsystem != nullptr ? RuntimeSubsystem->ResolveSpawnedPawnByVehicleName(VehicleName) : nullptr;
}

void UCityRuntimeValidationSubsystem::TrackPedestrian(const FString& PedId, ECityTrackedObjectKind Kind, const FString& GroupId)
{
	for (FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (TrackedObject.Id == PedId)
		{
			TrackedObject.Kind = Kind;
			TrackedObject.GroupId = GroupId;
			TrackedObject.LogicalAssetId = TEXT("pedestrian.cityops.basic.v1");
			TrackedObject.EntityId = PedId;
			TrackedObject.Actor = ResolvePedestrianActor(PedId);
			return;
		}
	}

	FCityTrackedRuntimeObject& TrackedObject = TrackedObjects.AddDefaulted_GetRef();
	TrackedObject.Id = PedId;
	TrackedObject.Kind = Kind;
	TrackedObject.GroupId = GroupId;
	TrackedObject.LogicalAssetId = TEXT("pedestrian.cityops.basic.v1");
	TrackedObject.EntityId = PedId;
	TrackedObject.Actor = ResolvePedestrianActor(PedId);
}

void UCityRuntimeValidationSubsystem::TrackAsset(const FString& AssetId, const FString& LogicalAssetId)
{
	for (FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (TrackedObject.Id == AssetId)
		{
			TrackedObject.Kind = ECityTrackedObjectKind::Asset;
			TrackedObject.LogicalAssetId = LogicalAssetId;
			TrackedObject.EntityId = AssetId;
			TrackedObject.Actor = ResolveAssetActor(AssetId);
			return;
		}
	}

	FCityTrackedRuntimeObject& TrackedObject = TrackedObjects.AddDefaulted_GetRef();
	TrackedObject.Id = AssetId;
	TrackedObject.Kind = ECityTrackedObjectKind::Asset;
	TrackedObject.LogicalAssetId = LogicalAssetId;
	TrackedObject.EntityId = AssetId;
	TrackedObject.Actor = ResolveAssetActor(AssetId);
}

void UCityRuntimeValidationSubsystem::TrackRuntimeVehicle(const FString& VehicleName, const FString& LogicalAssetId)
{
	for (FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (TrackedObject.Id == VehicleName)
		{
			TrackedObject.Kind = ECityTrackedObjectKind::RuntimeVehicle;
			TrackedObject.LogicalAssetId = LogicalAssetId;
			TrackedObject.EntityId = VehicleName;
			TrackedObject.Actor = ResolveRuntimeVehicleActor(VehicleName);
			return;
		}
	}

	FCityTrackedRuntimeObject& TrackedObject = TrackedObjects.AddDefaulted_GetRef();
	TrackedObject.Id = VehicleName;
	TrackedObject.Kind = ECityTrackedObjectKind::RuntimeVehicle;
	TrackedObject.LogicalAssetId = LogicalAssetId;
	TrackedObject.EntityId = VehicleName;
	TrackedObject.Actor = ResolveRuntimeVehicleActor(VehicleName);
}

void UCityRuntimeValidationSubsystem::UntrackGroup(const FString& GroupId)
{
	TrackedObjects.RemoveAll([&GroupId](const FCityTrackedRuntimeObject& Item) { return Item.GroupId == GroupId; });
}

void UCityRuntimeValidationSubsystem::UpdateTrackedObjects()
{
	for (FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		UpdateTrackedObjectActor(TrackedObject);
		UpdateGroundingState(TrackedObject);
	}
	RefreshFeedbackLinks();
}

void UCityRuntimeValidationSubsystem::UpdateTrackedObjectActor(FCityTrackedRuntimeObject& TrackedObject)
{
	if (TrackedObject.Actor.IsValid())
	{
		return;
	}

	switch (TrackedObject.Kind)
	{
	case ECityTrackedObjectKind::Asset:
		TrackedObject.Actor = ResolveAssetActor(TrackedObject.Id);
		break;
	case ECityTrackedObjectKind::RuntimeVehicle:
		TrackedObject.Actor = ResolveRuntimeVehicleActor(TrackedObject.Id);
		break;
	default:
		TrackedObject.Actor = ResolvePedestrianActor(TrackedObject.Id);
		break;
	}
}

void UCityRuntimeValidationSubsystem::UpdateGroundingState(FCityTrackedRuntimeObject& TrackedObject)
{
	TrackedObject.bGrounded = false;
	TrackedObject.GroundingMessage = TEXT("actor missing");
	if (!TrackedObject.Actor.IsValid() || GetWorld() == nullptr)
	{
		return;
	}

	if (TrackedObject.Kind == ECityTrackedObjectKind::RuntimeVehicle)
	{
		TrackedObject.bGrounded = true;
		TrackedObject.GroundingMessage = TEXT("airsim-flight-controlled");
		return;
	}

	FVector Origin;
	FVector Extent;
	TrackedObject.Actor->GetActorBounds(false, Origin, Extent);
	FVector GroundPoint = Origin;
	if (!AeroGroundPlacement::TryProjectWorldPointToGround(GetWorld(), Origin, GroundPoint, nullptr, TrackedObject.Actor.Get()))
	{
		TrackedObject.GroundingMessage = TEXT("no ground hit");
		return;
	}

	const float BottomZ = Origin.Z - Extent.Z;
	const float GapCm = BottomZ - GroundPoint.Z;
	if (FMath::Abs(GapCm) <= GroundingToleranceCm)
	{
		TrackedObject.bGrounded = true;
		TrackedObject.GroundingMessage = FString::Printf(TEXT("grounded (gap %.1f cm)"), GapCm);
	}
	else if (GapCm > GroundingToleranceCm)
	{
		TrackedObject.GroundingMessage = FString::Printf(TEXT("hovering %.1f cm above ground"), GapCm);
	}
	else
	{
		TrackedObject.GroundingMessage = FString::Printf(TEXT("embedded %.1f cm below ground"), -GapCm);
	}
}

void UCityRuntimeValidationSubsystem::RefreshFeedbackLinks()
{
	for (FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		TrackedObject.bHasFeedback = false;
		TrackedObject.LastFeedbackType.Reset();
		for (int32 Index = RecentFeedbackEvents.Num() - 1; Index >= 0; --Index)
		{
			const FAeroFeedbackEvent& Event = RecentFeedbackEvents[Index];
			const bool bMatches = Event.SourceEntityId.Equals(TrackedObject.Id, ESearchCase::IgnoreCase) ||
				Event.OtherEntityId.Equals(TrackedObject.Id, ESearchCase::IgnoreCase) ||
				Event.SourceActorId.Equals(TrackedObject.Id, ESearchCase::IgnoreCase) ||
				Event.OtherActorId.Equals(TrackedObject.Id, ESearchCase::IgnoreCase);
			if (bMatches)
			{
				TrackedObject.bHasFeedback = true;
				TrackedObject.LastFeedbackType = Event.Type;
				break;
			}
		}
	}
}

FString UCityRuntimeValidationSubsystem::BuildFeedbackSummary(const FAeroFeedbackEvent& Event) const
{
	const FString SourceId = !Event.SourceEntityId.IsEmpty() ? Event.SourceEntityId : Event.SourceActorId;
	const FString OtherId = !Event.OtherEntityId.IsEmpty() ? Event.OtherEntityId : Event.OtherActorId;
	if (Event.Type.Equals(TEXT("collision"), ESearchCase::IgnoreCase))
	{
		return FString::Printf(TEXT("%s | %s -> %s | speed=%.2f m/s"), *Event.Type, *SourceId, *OtherId, Event.Collision.RelativeSpeedMps);
	}
	return FString::Printf(TEXT("%s | %s -> %s"), *Event.Type, *SourceId, *OtherId);
}

bool UCityRuntimeValidationSubsystem::BindHudRuntimeVehicle(
	AActor* Actor,
	const FString& EntityId,
	const FString& LogicalAssetId,
	const TArray<FString>& Tags,
	const FString& LabelClass)
{
	if (!IsValid(Actor))
	{
		LastErrorMessage = FString::Printf(TEXT("Cannot bind HUD runtime vehicle '%s': actor is invalid."), *EntityId);
		return false;
	}

	FAeroSemanticBindingData BindingData;
	BindingData.EntityId = EntityId.TrimStartAndEnd();
	BindingData.InstanceId = BindingData.EntityId;
	BindingData.LogicalAssetId = LogicalAssetId;
	BindingData.Tags = Tags;
	BindingData.LabelClass = LabelClass;
	BindingData.FeedbackMode = EAeroFeedbackMode::Hit;
	FAeroSemanticRuntimeHelpers::ApplySemanticBinding(Actor, BindingData);
	FAeroSemanticRuntimeHelpers::EnsureCollisionRelay(Actor);
	return true;
}

void UCityRuntimeValidationSubsystem::LoadContext()
{
	SetStepState(TEXT("Load Context"), ECityValidationStepState::Running, TEXT("Loading default map context."));
	if (LoadContextInternal(DemoMapId))
	{
		SetStepState(TEXT("Load Context"), ECityValidationStepState::Passed, TEXT("Context loaded."));
	}
	else
	{
		SetStepState(TEXT("Load Context"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::ClearDemo()
{
	RemoveAll();
	ResetDemoScheduling();
	ResetValidationState();
	SetOverallState(ECityValidationStepState::Idle, TEXT("Demo cleared."));
}

void UCityRuntimeValidationSubsystem::PollFeedbackNow()
{
	if (UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>())
	{
		TArray<FAeroFeedbackEvent> Events;
		int64 UptoTick = 0;
		int64 UptoFrameId = 0;
		FString EpisodeId;
		FeedbackSubsystem->PollAllFeedback(Events, UptoTick, UptoFrameId, EpisodeId);
		RecentFeedbackEvents.Reset();
		const int32 StartIndex = FMath::Max(0, Events.Num() - MaxRecentFeedbackEvents);
		for (int32 Index = StartIndex; Index < Events.Num(); ++Index)
		{
			RecentFeedbackEvents.Add(Events[Index]);
		}
		RefreshFeedbackLinks();
		SetStepState(TEXT("Poll Feedback"), RecentFeedbackEvents.Num() > 0 ? ECityValidationStepState::Passed : ECityValidationStepState::Failed, RecentFeedbackEvents.Num() > 0 ? TEXT("Feedback received.") : TEXT("No feedback events found."));
	}
}

void UCityRuntimeValidationSubsystem::RecheckGrounding()
{
	UpdateTrackedObjects();
	bool bAllGrounded = true;
	for (const FCityTrackedRuntimeObject& TrackedObject : TrackedObjects)
	{
		if (!TrackedObject.bGrounded)
		{
			bAllGrounded = false;
			break;
		}
	}
	SetStepState(TEXT("Recheck Grounding"), bAllGrounded ? ECityValidationStepState::Passed : ECityValidationStepState::Failed, bAllGrounded ? TEXT("All tracked objects grounded.") : TEXT("One or more tracked objects failed grounding."));
}

void UCityRuntimeValidationSubsystem::SpawnPed()
{
	SetStepState(TEXT("Spawn Ped"), ECityValidationStepState::Running, TEXT("Spawning validation pedestrian."));
	FTransform RoadAnchorWorld;
	FSumoNearestLaneSample RoadSample;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld, &RoadSample))
	{
		SetStepState(TEXT("Spawn Ped"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector SpawnWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoPedSpawnLocalOffsetCm);
	const float SpawnYawDeg = ResolveRoadRelativeYawDeg(RoadAnchorWorld, DemoPedYawOffsetDeg);
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnPed road anchored: lane='%s' spawn='%s' yaw=%.2f."),
		*RoadSample.LaneId,
		*SpawnWorldCm.ToString(),
		SpawnYawDeg);
	if (SpawnPedInternal(DemoPedId, SpawnWorldCm, SpawnYawDeg, NAME_None, false))
	{
		TrackPedestrian(DemoPedId, ECityTrackedObjectKind::Pedestrian);
		SetStepState(TEXT("Spawn Ped"), ECityValidationStepState::Passed, TEXT("Pedestrian spawned."));
	}
	else
	{
		SetStepState(TEXT("Spawn Ped"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::ObservePed()
{
	SetStepState(TEXT("Observe"), ECityValidationStepState::Running, TEXT("Playing observe animation."));
	if (ObservePedInternal(DemoPedId))
	{
		SetStepState(TEXT("Observe"), ECityValidationStepState::Passed, TEXT("Observe animation triggered."));
	}
	else
	{
		SetStepState(TEXT("Observe"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::PlayPedAnimation(const FString& AnimationAssetPath, const FString& Label)
{
	const FString StepName = FString::Printf(TEXT("Anim: %s"), *Label);
	SetStepState(StepName, ECityValidationStepState::Running, FString::Printf(TEXT("Playing '%s'."), *Label));
	if (PlayPedAnimationInternal(DemoPedId, AnimationAssetPath))
	{
		SetStepState(StepName, ECityValidationStepState::Passed, FString::Printf(TEXT("'%s' triggered."), *Label));
	}
	else
	{
		SetStepState(StepName, ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::CommitCross()
{
	SetStepState(TEXT("Commit Cross"), ECityValidationStepState::Running, TEXT("Committing pedestrian cross."));
	FTransform RoadAnchorWorld;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld))
	{
		SetStepState(TEXT("Commit Cross"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector CrossTargetWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoPedCrossTargetLocalOffsetCm);
	if (CommitCrossInternal(DemoPedId, CrossTargetWorldCm, 150.0f, false))
	{
		SetStepState(TEXT("Commit Cross"), ECityValidationStepState::Passed, TEXT("Cross command issued."));
	}
	else
	{
		SetStepState(TEXT("Commit Cross"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::SpawnCrowd()
{
	SetStepState(TEXT("Spawn Crowd"), ECityValidationStepState::Running, TEXT("Spawning crowd."));
	FTransform RoadAnchorWorld;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld))
	{
		SetStepState(TEXT("Spawn Crowd"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector CrowdOriginWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoCrowdSpawnLocalOffsetCm);
	if (SpawnCrowdInternal(DemoCrowdGroupId, 4, CrowdOriginWorldCm, false))
	{
		SetStepState(TEXT("Spawn Crowd"), ECityValidationStepState::Passed, TEXT("Crowd spawned."));
	}
	else
	{
		SetStepState(TEXT("Spawn Crowd"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::SpawnCone()
{
	SetStepState(TEXT("Spawn Cone"), ECityValidationStepState::Running, TEXT("Spawning cone."));
	FTransform RoadAnchorWorld;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld))
	{
		SetStepState(TEXT("Spawn Cone"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector SpawnWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoConeSpawnLocalOffsetCm);
	const float SpawnYawDeg = ResolveRoadRelativeYawDeg(RoadAnchorWorld, 0.0f);
	if (SpawnAssetInternal(DemoConeAssetId, DemoConeLogicalAssetId, SpawnWorldCm, SpawnYawDeg, true))
	{
		TrackAsset(DemoConeAssetId, DemoConeLogicalAssetId);
		SetStepState(TEXT("Spawn Cone"), ECityValidationStepState::Passed, TEXT("Cone spawned."));
	}
	else
	{
		SetStepState(TEXT("Spawn Cone"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::SpawnStreetLightPlaceholder()
{
	SetStepState(TEXT("Spawn StreetLight Placeholder"), ECityValidationStepState::Running, TEXT("Spawning streetlight placeholder."));
	FTransform RoadAnchorWorld;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld))
	{
		SetStepState(TEXT("Spawn StreetLight Placeholder"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector SpawnWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoStreetLightSpawnLocalOffsetCm);
	const float SpawnYawDeg = ResolveRoadRelativeYawDeg(RoadAnchorWorld, 180.0f);
	if (SpawnAssetInternal(DemoStreetLightAssetId, DemoStreetLightLogicalAssetId, SpawnWorldCm, SpawnYawDeg, true))
	{
		TrackAsset(DemoStreetLightAssetId, DemoStreetLightLogicalAssetId);
		SetStepState(TEXT("Spawn StreetLight Placeholder"), ECityValidationStepState::Passed, TEXT("Streetlight placeholder spawned."));
	}
	else
	{
		SetStepState(TEXT("Spawn StreetLight Placeholder"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::SpawnSceneVehicle()
{
	FTransform RoadAnchorWorld;
	FSumoNearestLaneSample RoadSample;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld, &RoadSample))
	{
		SetStepState(StepSpawnVehicle, ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector VehicleSpawnWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoVehicleSpawnLocalOffsetCm);
	const FRotator VehicleSpawnRotation(0.0f, ResolveRoadRelativeYawDeg(RoadAnchorWorld, DemoVehicleYawOffsetDeg), 0.0f);
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnSceneVehicle road anchored: lane='%s' spawn='%s' yaw=%.2f."),
		*RoadSample.LaneId,
		*VehicleSpawnWorldCm.ToString(),
		VehicleSpawnRotation.Yaw);

	SetStepState(StepSpawnVehicle, ECityValidationStepState::Running, TEXT("Spawning scene-sync vehicle proxy."));
	if (!ActiveHudVehicleName.IsEmpty())
	{
		RemoveSceneEntityInternal(ActiveHudVehicleName);
		TrackedObjects.RemoveAll([this](const FCityTrackedRuntimeObject& Item) { return Item.Id == ActiveHudVehicleName; });
		ActiveHudVehicleName.Reset();
	}

	if (!SpawnSceneVehicleInternal(DemoVehicleAssetId, DemoVehicleLogicalAssetId, VehicleSpawnWorldCm, VehicleSpawnRotation.Yaw))
	{
		SetStepState(StepSpawnVehicle, ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	ActiveHudVehicleName = DemoVehicleAssetId;
	TrackAsset(ActiveHudVehicleName, DemoVehicleLogicalAssetId);
	SetStepState(StepSpawnVehicle, ECityValidationStepState::Passed, FString::Printf(TEXT("Scene-sync vehicle '%s' spawned."), *ActiveHudVehicleName));
}

void UCityRuntimeValidationSubsystem::SpawnRuntimeUAV()
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FAeroRuntimeAirSimCapabilities Capabilities;
	RuntimeSubsystem->GetCurrentAirSimCapabilities(Capabilities);
	if (!Capabilities.bSupportsMultirotors)
	{
		SetStepUnavailable(TEXT("Spawn Runtime UAV"), FString::Printf(TEXT("Unavailable in SimMode '%s'."), *Capabilities.SimModeName));
		return;
	}

	FTransform RoadAnchorWorld;
	FSumoNearestLaneSample RoadSample;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld, &RoadSample))
	{
		SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FString UavName = FString::Printf(TEXT("%s.%03d"), DemoUavNamePrefix, ++HudUavSpawnSequence);
	const FVector UavSpawnWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoUavSpawnLocalOffsetCm);
	const FRotator UavSpawnRotation(0.0f, ResolveRoadRelativeYawDeg(RoadAnchorWorld, DemoUavYawOffsetDeg), 0.0f);
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation SpawnRuntimeUAV road anchored: lane='%s' spawn='%s' yaw=%.2f."),
		*RoadSample.LaneId,
		*UavSpawnWorldCm.ToString(),
		UavSpawnRotation.Yaw);

	SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Running, TEXT("Spawning AirSim runtime UAV."));
	FString Error;
	if (!ActiveHudUavName.IsEmpty())
	{
		RuntimeSubsystem->RemoveRuntimeVehicle(ActiveHudUavName, Error);
		TrackedObjects.RemoveAll([this](const FCityTrackedRuntimeObject& Item) { return Item.Id == ActiveHudUavName; });
		ActiveHudUavName.Reset();
	}

	if (!RuntimeSubsystem->CreateRuntimeMultirotor(UavName, UavSpawnWorldCm, UavSpawnRotation, Error))
	{
		LastErrorMessage = Error;
		SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	AActor* UavActor = RuntimeSubsystem->ResolveSpawnedPawnByVehicleName(UavName);
	if (!BindHudRuntimeVehicle(
			UavActor,
			UavName,
			DemoUavRuntimeLogicalAssetId,
			{TEXT("uav"), TEXT("airsim"), TEXT("hud")},
			TEXT("uav")))
	{
		SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	ActiveHudUavName = UavName;
	TrackRuntimeVehicle(UavName, DemoUavRuntimeLogicalAssetId);
	SetStepState(TEXT("Spawn Runtime UAV"), ECityValidationStepState::Passed, FString::Printf(TEXT("AirSim UAV '%s' spawned."), *UavName));
}

void UCityRuntimeValidationSubsystem::MoveSceneVehicle()
{
	if (ActiveHudVehicleName.IsEmpty())
	{
		LastErrorMessage = TEXT("No active HUD scene-sync vehicle is available.");
		SetStepState(StepMoveVehicle, ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FTransform RoadAnchorWorld;
	FSumoNearestLaneSample RoadSample;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld, &RoadSample))
	{
		SetStepState(StepMoveVehicle, ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	const FVector VehicleTargetWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoVehicleMoveLocalOffsetCm);
	const FRotator VehicleTargetRotation(0.0f, ResolveRoadRelativeYawDeg(RoadAnchorWorld, DemoVehicleYawOffsetDeg), 0.0f);
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation MoveSceneVehicle road anchored: lane='%s' target='%s' yaw=%.2f."),
		*RoadSample.LaneId,
		*VehicleTargetWorldCm.ToString(),
		VehicleTargetRotation.Yaw);

	SetStepState(StepMoveVehicle, ECityValidationStepState::Running, TEXT("Moving scene-sync vehicle proxy."));
	if (MoveSceneVehicleInternal(ActiveHudVehicleName, VehicleTargetWorldCm, VehicleTargetRotation.Yaw))
	{
		SetStepState(StepMoveVehicle, ECityValidationStepState::Passed, TEXT("Scene-sync vehicle moved."));
	}
	else
	{
		SetStepState(StepMoveVehicle, ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::MoveRuntimeUAV()
{
	UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr;
	if (RuntimeSubsystem == nullptr)
	{
		LastErrorMessage = TEXT("AeroRuntimeOrchestrationSubsystem unavailable.");
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FAeroRuntimeAirSimCapabilities Capabilities;
	RuntimeSubsystem->GetCurrentAirSimCapabilities(Capabilities);
	if (!Capabilities.bSupportsMultirotors)
	{
		SetStepUnavailable(TEXT("Move UAV"), FString::Printf(TEXT("Unavailable in SimMode '%s'."), *Capabilities.SimModeName));
		return;
	}

	if (ActiveHudUavName.IsEmpty())
	{
		LastErrorMessage = TEXT("No active HUD AirSim UAV is available.");
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FVector CurrentUavWorldCm = FVector::ZeroVector;
	FRotator CurrentUavRotation = FRotator::ZeroRotator;
	FString Error;
	if (!RuntimeSubsystem->GetVehiclePose(ActiveHudUavName, CurrentUavWorldCm, CurrentUavRotation, Error))
	{
		LastErrorMessage = Error;
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FTransform RoadAnchorWorld;
	FSumoNearestLaneSample RoadSample;
	if (!ResolveDemoRoadAnchor(RoadAnchorWorld, &RoadSample))
	{
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
		return;
	}

	FVector UavTargetWorldCm = ApplyRoadLocalOffset(RoadAnchorWorld, DemoUavMoveLocalOffsetCm);
	UavTargetWorldCm.Z = CurrentUavWorldCm.Z;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("CityRuntimeValidation MoveRuntimeUAV road anchored: lane='%s' target='%s' preserved_z=%.2f."),
		*RoadSample.LaneId,
		*UavTargetWorldCm.ToString(),
		CurrentUavWorldCm.Z);

	SetStepState(TEXT("Move UAV"), ECityValidationStepState::Running, TEXT("Starting async AirSim UAV move."));
	if (RuntimeSubsystem->MoveMultirotorToPosition(ActiveHudUavName, UavTargetWorldCm, DemoUavMoveVelocityMps, Error))
	{
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Running, TEXT("AirSim UAV move is running asynchronously."));
	}
	else
	{
		LastErrorMessage = Error;
		SetStepState(TEXT("Move UAV"), ECityValidationStepState::Failed, LastErrorMessage);
	}
}

void UCityRuntimeValidationSubsystem::RemoveAll()
{
	ReleasePedInternal(DemoPedId);
	ClearCrowdInternal(DemoCrowdGroupId);
	RemoveAssetInternal(DemoConeAssetId);
	RemoveAssetInternal(DemoStreetLightAssetId);
	if (!ActiveHudVehicleName.IsEmpty())
	{
		RemoveSceneEntityInternal(ActiveHudVehicleName);
	}
	if (UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr)
	{
		FString Error;
		if (!ActiveHudUavName.IsEmpty())
		{
			RuntimeSubsystem->RemoveRuntimeVehicle(ActiveHudUavName, Error);
		}
	}
	TrackedObjects.Reset();
	RecentFeedbackEvents.Reset();
	ActiveHudVehicleName.Reset();
	ActiveHudUavName.Reset();
}

void UCityRuntimeValidationSubsystem::RunFullDemo()
{
	ClearDemo();
	SetOverallState(ECityValidationStepState::Running, TEXT("Running full demo."));
	bDemoRunning = true;

	FAeroRuntimeAirSimCapabilities Capabilities;
	if (UAeroRuntimeOrchestrationSubsystem* RuntimeSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroRuntimeOrchestrationSubsystem>() : nullptr)
	{
		RuntimeSubsystem->GetCurrentAirSimCapabilities(Capabilities);
	}

	if (!Capabilities.bSupportsMultirotors)
	{
		SetStepUnavailable(TEXT("Spawn Runtime UAV"), FString::Printf(TEXT("Unavailable in SimMode '%s'."), *Capabilities.SimModeName));
		SetStepUnavailable(TEXT("Move UAV"), FString::Printf(TEXT("Unavailable in SimMode '%s'."), *Capabilities.SimModeName));
	}

	ScheduleAction(0.0, [this]() { LoadContext(); });
	ScheduleAction(0.15, [this]() { SpawnPed(); });
	ScheduleAction(0.30, [this]() {
		SetStepState(TEXT("Set Variant"), ECityValidationStepState::Running, TEXT("Switching variant."));
		if (SetPedVariantInternal(DemoPedId, FName(DemoPedVariantId)))
		{
			SetStepState(TEXT("Set Variant"), ECityValidationStepState::Passed, DemoPedVariantId);
		}
		else
		{
			SetStepState(TEXT("Set Variant"), ECityValidationStepState::Failed, LastErrorMessage);
		}
	});
	ScheduleAction(0.45, [this]() { ObservePed(); });
	ScheduleAction(2.20, [this]() { CommitCross(); });
	ScheduleAction(2.35, [this]() { SpawnCrowd(); });
	ScheduleAction(2.50, [this]() { SpawnCone(); });
	ScheduleAction(2.65, [this]() { SpawnStreetLightPlaceholder(); });
	ScheduleAction(2.80, [this]() { SpawnSceneVehicle(); });
	ScheduleAction(3.20, [this]() { MoveSceneVehicle(); });
	if (Capabilities.bSupportsMultirotors)
	{
		ScheduleAction(2.95, [this]() { SpawnRuntimeUAV(); });
		ScheduleAction(3.35, [this]() { MoveRuntimeUAV(); });
	}
	ScheduleAction(4.30, [this]() { PollFeedbackNow(); });
	ScheduleAction(4.45, [this]() {
		RecheckGrounding();
		bool bHasRunningStep = false;
		for (const FCityValidationStepResult& StepResult : StepResults)
		{
			if (StepResult.State == ECityValidationStepState::Running)
			{
				bHasRunningStep = true;
				break;
			}
		}
		if (OverallState != ECityValidationStepState::Failed && !bHasRunningStep)
		{
			SetOverallState(ECityValidationStepState::Passed, TEXT("Full demo finished."));
		}
		bDemoRunning = false;
	});
}
