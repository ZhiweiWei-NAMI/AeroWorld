#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"

class UCityRuntimeValidationSubsystem;

class SCityRuntimeValidationPanel : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(SCityRuntimeValidationPanel)
	{
	}

		SLATE_ARGUMENT(TWeakObjectPtr<UCityRuntimeValidationSubsystem>, ValidationSubsystem)

	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

private:
	FReply HandleLoadContextClicked();
	FReply HandleRunFullDemoClicked();
	FReply HandleClearDemoClicked();
	FReply HandleToggleDetailsClicked();
	FReply HandlePollFeedbackClicked();
	FReply HandleRecheckGroundingClicked();
	FReply HandleSpawnPedClicked();
	FReply HandleObserveClicked();
	FReply HandleCommitCrossClicked();
	FReply HandleSpawnCrowdClicked();
	FReply HandleSpawnConeClicked();
	FReply HandleSpawnStreetLightClicked();
	FReply HandleSpawnVehicleClicked();
	FReply HandleSpawnUAVClicked();
	FReply HandleMoveVehicleClicked();
	FReply HandleMoveUAVClicked();
	FReply HandleRemoveAllClicked();
	FReply HandleAnimTalkingClicked();
	FReply HandleAnimYellingClicked();
	FReply HandleAnimPhonePacingClicked();
	FReply HandleAnimLookingAroundClicked();
	FReply HandleAnimHitReactionClicked();
	FReply HandleAnimFallFlatClicked();
	FReply HandleAnimRunningClicked();
	FReply HandleAnimHappyIdleClicked();
	FReply HandleAnimSadIdleClicked();
	FReply HandleAnimRappingClicked();

	FText GetHeaderText() const;
	FSlateColor GetHeaderColor() const;
	FText GetToggleDetailsText() const;
	EVisibility GetDetailsVisibility() const;
	FText GetStatusSummaryText() const;
	FText GetCapabilitiesText() const;
	FText GetStepResultsText() const;
	FText GetTrackedCountsText() const;
	FText GetFeedbackText() const;
	FText GetPassedObjectsText() const;
	FText GetPendingObjectsText() const;
	FText GetFailedObjectsText() const;

	TWeakObjectPtr<UCityRuntimeValidationSubsystem> ValidationSubsystem;
	bool bDetailsExpanded = false;
};
