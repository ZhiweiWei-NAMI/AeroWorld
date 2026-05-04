#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroSemanticTypes.h"
#include "AeroFeedbackSubsystem.generated.h"

UCLASS()
class AEROSEMANTICRUNTIME_API UAeroFeedbackSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Deinitialize() override;

	void SetFrameContext(const FAeroFrameContext& InFrameContext);
	const FAeroFrameContext& GetCurrentFrameContext() const;

	void SetWorldOriginCm(const FVector& InWorldOriginCm);
	const FVector& GetWorldOriginCm() const;
	FVector WorldCmToEnuM(const FVector& WorldCm) const;

	void EnqueueFeedback(FAeroFeedbackEvent Event);
	void PollFeedbackSinceTick(int64 SinceTick, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const;
	void PollFeedbackSinceFrame(int64 SinceFrameId, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const;
	void PollAllFeedback(TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const;
	void ResetEpisode(const FString& InEpisodeId);

private:
	void TrimBufferIfNeeded();
	void PollFeedbackInternal(int64 SinceValue, bool bUseTick, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const;
	bool ShouldSuppressDuplicate(const FAeroFeedbackEvent& Event);

private:
	UPROPERTY()
	int32 MaxBufferedEvents = 4096;

	FAeroFrameContext CurrentFrameContext;
	FVector WorldOriginCm = FVector::ZeroVector;
	TArray<FAeroFeedbackEvent> FeedbackEvents;
	TSet<FString> RecentEventKeys;
	uint64 EventSequence = 0;
};
