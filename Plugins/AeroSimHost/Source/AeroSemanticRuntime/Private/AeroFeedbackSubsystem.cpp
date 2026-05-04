#include "AeroFeedbackSubsystem.h"

#include "Engine/World.h"

namespace
{
FString BuildFeedbackDedupKey(const FAeroFeedbackEvent& Event)
{
	const FString EventType = Event.Type.TrimStartAndEnd().ToLower();
	if (EventType.IsEmpty())
	{
		return FString();
	}

	const FString SourceKey = !Event.SourceEntityId.TrimStartAndEnd().IsEmpty() ? Event.SourceEntityId : Event.SourceActorId;
	const FString OtherKey = !Event.OtherEntityId.TrimStartAndEnd().IsEmpty() ? Event.OtherEntityId : Event.OtherActorId;
	if (SourceKey.TrimStartAndEnd().IsEmpty() && OtherKey.TrimStartAndEnd().IsEmpty())
	{
		return FString();
	}

	if (EventType.Equals(TEXT("collision"), ESearchCase::CaseSensitive))
	{
		FString First = SourceKey;
		FString Second = OtherKey;
		if (Second < First)
		{
			Swap(First, Second);
		}

		return FString::Printf(
			TEXT("%s|%s|%lld|%lld|%lld|%s|%s"),
			*Event.EpisodeId,
			*EventType,
			static_cast<long long>(Event.Tick),
			static_cast<long long>(Event.FrameId),
			static_cast<long long>(Event.SampleSeq),
			*First,
			*Second);
	}

	return FString::Printf(
		TEXT("%s|%s|%lld|%lld|%lld|%s|%s|%s|%s"),
		*Event.EpisodeId,
		*EventType,
		static_cast<long long>(Event.Tick),
		static_cast<long long>(Event.FrameId),
		static_cast<long long>(Event.SampleSeq),
		*SourceKey,
		*OtherKey,
		*Event.Overlap.WorldLayerType,
		*Event.Overlap.ZoneKind);
}
}

bool UAeroFeedbackSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroFeedbackSubsystem::Deinitialize()
{
	FeedbackEvents.Reset();
	RecentEventKeys.Reset();
	CurrentFrameContext = FAeroFrameContext();
	WorldOriginCm = FVector::ZeroVector;
	EventSequence = 0;
	Super::Deinitialize();
}

void UAeroFeedbackSubsystem::SetFrameContext(const FAeroFrameContext& InFrameContext)
{
	if (!CurrentFrameContext.EpisodeId.IsEmpty() && !InFrameContext.EpisodeId.IsEmpty() && CurrentFrameContext.EpisodeId != InFrameContext.EpisodeId)
	{
		ResetEpisode(InFrameContext.EpisodeId);
	}

	CurrentFrameContext = InFrameContext;
}

const FAeroFrameContext& UAeroFeedbackSubsystem::GetCurrentFrameContext() const
{
	return CurrentFrameContext;
}

void UAeroFeedbackSubsystem::SetWorldOriginCm(const FVector& InWorldOriginCm)
{
	WorldOriginCm = InWorldOriginCm;
}

const FVector& UAeroFeedbackSubsystem::GetWorldOriginCm() const
{
	return WorldOriginCm;
}

FVector UAeroFeedbackSubsystem::WorldCmToEnuM(const FVector& WorldCm) const
{
	return (WorldCm - WorldOriginCm) / 100.0;
}

void UAeroFeedbackSubsystem::EnqueueFeedback(FAeroFeedbackEvent Event)
{
	if (Event.EventId.TrimStartAndEnd().IsEmpty())
	{
		Event.EventId = FString::Printf(TEXT("evt_%06llu"), static_cast<unsigned long long>(++EventSequence));
	}

	if (Event.Tick == 0)
	{
		Event.Tick = CurrentFrameContext.Tick;
	}
	if (Event.FrameId == 0)
	{
		Event.FrameId = CurrentFrameContext.FrameId;
	}
	if (Event.EpisodeId.IsEmpty())
	{
		Event.EpisodeId = CurrentFrameContext.EpisodeId;
	}
	if (Event.SampleSeq == INDEX_NONE)
	{
		Event.SampleSeq = CurrentFrameContext.SampleSeq;
	}
	if (Event.SimTimeS == 0.0)
	{
		Event.SimTimeS = CurrentFrameContext.SimTimeS;
	}

	if (ShouldSuppressDuplicate(Event))
	{
		return;
	}

	FeedbackEvents.Add(MoveTemp(Event));
	TrimBufferIfNeeded();
}

void UAeroFeedbackSubsystem::PollFeedbackSinceTick(int64 SinceTick, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const
{
	PollFeedbackInternal(SinceTick, true, OutEvents, OutUptoTick, OutUptoFrameId, OutEpisodeId);
}

void UAeroFeedbackSubsystem::PollFeedbackSinceFrame(int64 SinceFrameId, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const
{
	PollFeedbackInternal(SinceFrameId, false, OutEvents, OutUptoTick, OutUptoFrameId, OutEpisodeId);
}

void UAeroFeedbackSubsystem::PollAllFeedback(TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const
{
	OutEvents = FeedbackEvents;
	OutUptoTick = CurrentFrameContext.Tick;
	OutUptoFrameId = CurrentFrameContext.FrameId;
	OutEpisodeId = CurrentFrameContext.EpisodeId;
}

void UAeroFeedbackSubsystem::ResetEpisode(const FString& InEpisodeId)
{
	FeedbackEvents.Reset();
	RecentEventKeys.Reset();
	EventSequence = 0;
	CurrentFrameContext = FAeroFrameContext();
	CurrentFrameContext.EpisodeId = InEpisodeId;
}

void UAeroFeedbackSubsystem::TrimBufferIfNeeded()
{
	if (MaxBufferedEvents <= 0 || FeedbackEvents.Num() <= MaxBufferedEvents)
	{
		return;
	}

	const int32 NumToRemove = FeedbackEvents.Num() - MaxBufferedEvents;
	FeedbackEvents.RemoveAt(0, NumToRemove, false);
}

void UAeroFeedbackSubsystem::PollFeedbackInternal(int64 SinceValue, bool bUseTick, TArray<FAeroFeedbackEvent>& OutEvents, int64& OutUptoTick, int64& OutUptoFrameId, FString& OutEpisodeId) const
{
	OutEvents.Reset();
	for (const FAeroFeedbackEvent& Event : FeedbackEvents)
	{
		const int64 CandidateValue = bUseTick ? Event.Tick : Event.FrameId;
		if (CandidateValue > SinceValue)
		{
			OutEvents.Add(Event);
		}
	}

	OutUptoTick = CurrentFrameContext.Tick;
	OutUptoFrameId = CurrentFrameContext.FrameId;
	OutEpisodeId = CurrentFrameContext.EpisodeId;
}

bool UAeroFeedbackSubsystem::ShouldSuppressDuplicate(const FAeroFeedbackEvent& Event)
{
	const FString DedupKey = BuildFeedbackDedupKey(Event);
	if (DedupKey.IsEmpty())
	{
		return false;
	}

	if (RecentEventKeys.Contains(DedupKey))
	{
		return true;
	}

	RecentEventKeys.Add(DedupKey);
	return false;
}
