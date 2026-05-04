#include "SCityRuntimeValidationPanel.h"

#include "CityRuntimeValidationSubsystem.h"
#include "Styling/CoreStyle.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SOverlay.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SUniformGridPanel.h"
#include "Widgets/Text/STextBlock.h"

namespace
{
TSharedRef<SWidget> MakeActionButton(const FString& Label, const FOnClicked& OnClicked)
{
	return SNew(SButton)
		.Text(FText::FromString(Label))
		.HAlign(HAlign_Center)
		.VAlign(VAlign_Center)
		.OnClicked(OnClicked);
}

TSharedRef<SWidget> MakeSectionHeader(const FString& Label)
{
	return SNew(STextBlock)
		.Text(FText::FromString(Label))
		.Font(FCoreStyle::GetDefaultFontStyle("Bold", 12));
}
}

void SCityRuntimeValidationPanel::Construct(const FArguments& InArgs)
{
	ValidationSubsystem = InArgs._ValidationSubsystem;

	ChildSlot
	[
		SNew(SOverlay)
		.Visibility(EVisibility::SelfHitTestInvisible)
		+ SOverlay::Slot()
		.HAlign(HAlign_Right)
		.VAlign(VAlign_Top)
		.Padding(FMargin(16.0f))
		[
			SNew(SBorder)
			.BorderImage(FCoreStyle::Get().GetBrush("ToolPanel.GroupBorder"))
			.Padding(FMargin(8.0f))
			.Visibility(EVisibility::Visible)
			[
				SNew(SBox)
				.WidthOverride(320.0f)
				.HeightOverride(180.0f)
				[
					SNew(SScrollBox)
					+ SScrollBox::Slot()
					[
						SNew(SVerticalBox)
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0.0f, 0.0f, 0.0f, 6.0f)
						[
							SNew(SHorizontalBox)
							+ SHorizontalBox::Slot()
							.FillWidth(1.0f)
							.VAlign(VAlign_Center)
							[
								SNew(STextBlock)
								.Text(this, &SCityRuntimeValidationPanel::GetHeaderText)
								.ColorAndOpacity(this, &SCityRuntimeValidationPanel::GetHeaderColor)
								.Font(FCoreStyle::GetDefaultFontStyle("Bold", 14))
							]
							+ SHorizontalBox::Slot()
							.AutoWidth()
							.Padding(6.0f, 0.0f, 0.0f, 0.0f)
							[
								SNew(SButton)
								.Text(this, &SCityRuntimeValidationPanel::GetToggleDetailsText)
								.OnClicked(FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleToggleDetailsClicked))
							]
						]
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0.0f, 0.0f, 0.0f, 8.0f)
						[
							SNew(STextBlock)
							.Text(this, &SCityRuntimeValidationPanel::GetStatusSummaryText)
							.AutoWrapText(true)
						]
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0.0f, 0.0f, 0.0f, 8.0f)
						[
							SNew(STextBlock)
							.Text(this, &SCityRuntimeValidationPanel::GetCapabilitiesText)
							.AutoWrapText(true)
							.Visibility(this, &SCityRuntimeValidationPanel::GetDetailsVisibility)
						]
						+ SVerticalBox::Slot()
						.AutoHeight()
						[
							SNew(SUniformGridPanel)
							.SlotPadding(FMargin(2.0f))
							+ SUniformGridPanel::Slot(0, 0)[MakeActionButton(TEXT("Load Context"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleLoadContextClicked))]
							+ SUniformGridPanel::Slot(1, 0)[MakeActionButton(TEXT("Spawn Ped"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnPedClicked))]
							+ SUniformGridPanel::Slot(0, 1)[MakeActionButton(TEXT("Spawn Crowd"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnCrowdClicked))]
							+ SUniformGridPanel::Slot(1, 1)[MakeActionButton(TEXT("Clear Demo"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleClearDemoClicked))]
						]
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0.0f, 8.0f, 0.0f, 0.0f)
						[
							SNew(SVerticalBox)
							.Visibility(this, &SCityRuntimeValidationPanel::GetDetailsVisibility)
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 0.0f, 0.0f, 8.0f)[SNew(SSeparator)]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 0.0f, 0.0f, 8.0f)[MakeSectionHeader(TEXT("Demo Flow"))]
							+ SVerticalBox::Slot().AutoHeight()
							[
								SNew(SUniformGridPanel)
								.SlotPadding(FMargin(2.0f))
								+ SUniformGridPanel::Slot(0, 0)[MakeActionButton(TEXT("Run Full Demo"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleRunFullDemoClicked))]
								+ SUniformGridPanel::Slot(1, 0)[MakeActionButton(TEXT("Poll Feedback"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandlePollFeedbackClicked))]
								+ SUniformGridPanel::Slot(0, 1)[MakeActionButton(TEXT("Recheck Grounding"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleRecheckGroundingClicked))]
							]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 10.0f, 0.0f, 8.0f)[SNew(SSeparator)]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 0.0f, 0.0f, 8.0f)[MakeSectionHeader(TEXT("Key Actions"))]
							+ SVerticalBox::Slot().AutoHeight()
							[
								SNew(SUniformGridPanel)
								.SlotPadding(FMargin(2.0f))
								+ SUniformGridPanel::Slot(0, 0)[MakeActionButton(TEXT("Observe"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleObserveClicked))]
								+ SUniformGridPanel::Slot(1, 0)[MakeActionButton(TEXT("Commit Cross"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleCommitCrossClicked))]
								+ SUniformGridPanel::Slot(0, 1)[MakeActionButton(TEXT("Spawn Cone"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnConeClicked))]
								+ SUniformGridPanel::Slot(1, 1)[MakeActionButton(TEXT("Spawn StreetLight"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnStreetLightClicked))]
								+ SUniformGridPanel::Slot(0, 2)[MakeActionButton(TEXT("Spawn Scene Vehicle"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnVehicleClicked))]
								+ SUniformGridPanel::Slot(1, 2)[MakeActionButton(TEXT("Spawn UAV"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleSpawnUAVClicked))]
								+ SUniformGridPanel::Slot(0, 3)[MakeActionButton(TEXT("Move Scene Vehicle"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleMoveVehicleClicked))]
								+ SUniformGridPanel::Slot(1, 3)[MakeActionButton(TEXT("Move UAV"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleMoveUAVClicked))]
								+ SUniformGridPanel::Slot(0, 4)[MakeActionButton(TEXT("Remove All"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleRemoveAllClicked))]
							]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 10.0f, 0.0f, 8.0f)[SNew(SSeparator)]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 0.0f, 0.0f, 8.0f)[MakeSectionHeader(TEXT("Ped Animations"))]
							+ SVerticalBox::Slot().AutoHeight()
							[
								SNew(SUniformGridPanel)
								.SlotPadding(FMargin(2.0f))
								+ SUniformGridPanel::Slot(0, 0)[MakeActionButton(TEXT("Talking"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimTalkingClicked))]
								+ SUniformGridPanel::Slot(1, 0)[MakeActionButton(TEXT("Yelling"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimYellingClicked))]
								+ SUniformGridPanel::Slot(0, 1)[MakeActionButton(TEXT("Phone Pacing"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimPhonePacingClicked))]
								+ SUniformGridPanel::Slot(1, 1)[MakeActionButton(TEXT("Looking Around"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimLookingAroundClicked))]
								+ SUniformGridPanel::Slot(0, 2)[MakeActionButton(TEXT("Hit Reaction"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimHitReactionClicked))]
								+ SUniformGridPanel::Slot(1, 2)[MakeActionButton(TEXT("Fall Flat"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimFallFlatClicked))]
								+ SUniformGridPanel::Slot(0, 3)[MakeActionButton(TEXT("Running"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimRunningClicked))]
								+ SUniformGridPanel::Slot(1, 3)[MakeActionButton(TEXT("Happy Idle"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimHappyIdleClicked))]
								+ SUniformGridPanel::Slot(0, 4)[MakeActionButton(TEXT("Sad Idle"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimSadIdleClicked))]
								+ SUniformGridPanel::Slot(1, 4)[MakeActionButton(TEXT("Rapping"), FOnClicked::CreateSP(this, &SCityRuntimeValidationPanel::HandleAnimRappingClicked))]
							]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 10.0f, 0.0f, 8.0f)[SNew(SSeparator)]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Counts"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 8.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetTrackedCountsText).AutoWrapText(true)]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Step Results"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 8.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetStepResultsText).AutoWrapText(true)]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Recent Feedback"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 8.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetFeedbackText).AutoWrapText(true)]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Grounded / Passed"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 8.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetPassedObjectsText).AutoWrapText(true).ColorAndOpacity(FLinearColor(0.35f, 0.95f, 0.45f))]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Pending"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 8.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetPendingObjectsText).AutoWrapText(true).ColorAndOpacity(FLinearColor(0.95f, 0.85f, 0.25f))]
							+ SVerticalBox::Slot().AutoHeight()[MakeSectionHeader(TEXT("Failed"))]
							+ SVerticalBox::Slot().AutoHeight().Padding(0.0f, 2.0f, 0.0f, 0.0f)[SNew(STextBlock).Text(this, &SCityRuntimeValidationPanel::GetFailedObjectsText).AutoWrapText(true).ColorAndOpacity(FLinearColor(0.95f, 0.35f, 0.35f))]
						]
					]
				]
			]
		]
	];
}

#define SUBSYSTEM_CALL(FuncName) \
	if (UCityRuntimeValidationSubsystem* Subsystem = ValidationSubsystem.Get()) { Subsystem->FuncName(); } \
	return FReply::Handled()

FReply SCityRuntimeValidationPanel::HandleLoadContextClicked() { SUBSYSTEM_CALL(LoadContext); }
FReply SCityRuntimeValidationPanel::HandleRunFullDemoClicked() { SUBSYSTEM_CALL(RunFullDemo); }
FReply SCityRuntimeValidationPanel::HandleClearDemoClicked() { SUBSYSTEM_CALL(ClearDemo); }
FReply SCityRuntimeValidationPanel::HandleToggleDetailsClicked()
{
	bDetailsExpanded = !bDetailsExpanded;
	return FReply::Handled();
}
FReply SCityRuntimeValidationPanel::HandlePollFeedbackClicked() { SUBSYSTEM_CALL(PollFeedbackNow); }
FReply SCityRuntimeValidationPanel::HandleRecheckGroundingClicked() { SUBSYSTEM_CALL(RecheckGrounding); }
FReply SCityRuntimeValidationPanel::HandleSpawnPedClicked() { SUBSYSTEM_CALL(SpawnPed); }
FReply SCityRuntimeValidationPanel::HandleObserveClicked() { SUBSYSTEM_CALL(ObservePed); }
FReply SCityRuntimeValidationPanel::HandleCommitCrossClicked() { SUBSYSTEM_CALL(CommitCross); }
FReply SCityRuntimeValidationPanel::HandleSpawnCrowdClicked() { SUBSYSTEM_CALL(SpawnCrowd); }
FReply SCityRuntimeValidationPanel::HandleSpawnConeClicked() { SUBSYSTEM_CALL(SpawnCone); }
FReply SCityRuntimeValidationPanel::HandleSpawnStreetLightClicked() { SUBSYSTEM_CALL(SpawnStreetLightPlaceholder); }
FReply SCityRuntimeValidationPanel::HandleSpawnVehicleClicked() { SUBSYSTEM_CALL(SpawnSceneVehicle); }
FReply SCityRuntimeValidationPanel::HandleSpawnUAVClicked() { SUBSYSTEM_CALL(SpawnRuntimeUAV); }
FReply SCityRuntimeValidationPanel::HandleMoveVehicleClicked() { SUBSYSTEM_CALL(MoveSceneVehicle); }
FReply SCityRuntimeValidationPanel::HandleMoveUAVClicked() { SUBSYSTEM_CALL(MoveRuntimeUAV); }
FReply SCityRuntimeValidationPanel::HandleRemoveAllClicked() { SUBSYSTEM_CALL(RemoveAll); }

#undef SUBSYSTEM_CALL

#define ANIM_HANDLER(HandlerName, AssetPath, Label) \
	FReply SCityRuntimeValidationPanel::HandlerName() \
	{ \
		if (UCityRuntimeValidationSubsystem* Subsystem = ValidationSubsystem.Get()) \
		{ \
			Subsystem->PlayPedAnimation(TEXT(AssetPath), TEXT(Label)); \
		} \
		return FReply::Handled(); \
	}

ANIM_HANDLER(HandleAnimTalkingClicked, "/Game/MixamoAssets/Animations/Talking.Talking", "Talking")
ANIM_HANDLER(HandleAnimYellingClicked, "/Game/MixamoAssets/Animations/Yelling.Yelling", "Yelling")
ANIM_HANDLER(HandleAnimPhonePacingClicked, "/Game/MixamoAssets/Animations/Talking_Phone_Pacing.Talking_Phone_Pacing", "Phone Pacing")
ANIM_HANDLER(HandleAnimLookingAroundClicked, "/Game/MixamoAssets/Animations/Looking_Around.Looking_Around", "Looking Around")
ANIM_HANDLER(HandleAnimHitReactionClicked, "/Game/MixamoAssets/Animations/Hit_Reaction.Hit_Reaction", "Hit Reaction")
ANIM_HANDLER(HandleAnimFallFlatClicked, "/Game/MixamoAssets/Animations/Fall_Flat.Fall_Flat", "Fall Flat")
ANIM_HANDLER(HandleAnimRunningClicked, "/Game/MixamoAssets/Animations/Running.Running", "Running")
ANIM_HANDLER(HandleAnimHappyIdleClicked, "/Game/MixamoAssets/Animations/Happy_Idle.Happy_Idle", "Happy Idle")
ANIM_HANDLER(HandleAnimSadIdleClicked, "/Game/MixamoAssets/Animations/Sad_Idle.Sad_Idle", "Sad Idle")
ANIM_HANDLER(HandleAnimRappingClicked, "/Game/MixamoAssets/Animations/Rapping.Rapping", "Rapping")

#undef ANIM_HANDLER

FText SCityRuntimeValidationPanel::GetHeaderText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetHeaderText() : FString(TEXT("PIE Runtime Validation"))); }
FSlateColor SCityRuntimeValidationPanel::GetHeaderColor() const { return ValidationSubsystem.IsValid() ? ValidationSubsystem->GetHeaderColor() : FSlateColor(FLinearColor::White); }
FText SCityRuntimeValidationPanel::GetToggleDetailsText() const { return FText::FromString(bDetailsExpanded ? TEXT("Hide Details") : TEXT("Show Details")); }
EVisibility SCityRuntimeValidationPanel::GetDetailsVisibility() const { return bDetailsExpanded ? EVisibility::Visible : EVisibility::Collapsed; }
FText SCityRuntimeValidationPanel::GetStatusSummaryText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetStatusSummaryText() : FString()); }
FText SCityRuntimeValidationPanel::GetCapabilitiesText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetCapabilitiesText() : FString()); }
FText SCityRuntimeValidationPanel::GetStepResultsText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetStepResultsText() : FString()); }
FText SCityRuntimeValidationPanel::GetTrackedCountsText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetTrackedCountsText() : FString()); }
FText SCityRuntimeValidationPanel::GetFeedbackText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetFeedbackText() : FString()); }
FText SCityRuntimeValidationPanel::GetPassedObjectsText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetPassedObjectsText() : FString()); }
FText SCityRuntimeValidationPanel::GetPendingObjectsText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetPendingObjectsText() : FString()); }
FText SCityRuntimeValidationPanel::GetFailedObjectsText() const { return FText::FromString(ValidationSubsystem.IsValid() ? ValidationSubsystem->GetFailedObjectsText() : FString()); }
