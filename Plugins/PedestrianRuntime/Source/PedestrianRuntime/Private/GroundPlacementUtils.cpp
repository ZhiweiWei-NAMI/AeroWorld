#include "GroundPlacementUtils.h"

#include "AeroPedNavSemanticSubsystem.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

namespace
{
constexpr float GroundProjectionTraceHalfHeightCm = 250000.0f;
constexpr float GroundTraceLiftPreferenceCm = 2.0f;

bool IsPreferredGroundActor(const AActor* Actor)
{
	if (!IsValid(Actor))
	{
		return false;
	}

	static const FName PreferredGroundTags[] = {
		FName(TEXT("terrain")),
		FName(TEXT("ground")),
		FName(TEXT("road")),
		FName(TEXT("sidewalk")),
		FName(TEXT("landscape"))};
	for (const FName& Tag : PreferredGroundTags)
	{
		if (Actor->ActorHasTag(Tag))
		{
			return true;
		}
	}

	const FString ActorName = Actor->GetName();
	if (ActorName.Contains(TEXT("road"), ESearchCase::IgnoreCase) ||
		ActorName.Contains(TEXT("sidewalk"), ESearchCase::IgnoreCase) ||
		ActorName.Contains(TEXT("bridge"), ESearchCase::IgnoreCase) ||
		ActorName.Contains(TEXT("citybase"), ESearchCase::IgnoreCase))
	{
		return true;
	}

	return Actor->GetClass() != nullptr &&
		(Actor->GetClass()->GetName().Contains(TEXT("Landscape")) ||
		 Actor->GetClass()->GetName().Contains(TEXT("Road"), ESearchCase::IgnoreCase) ||
		 Actor->GetClass()->GetName().Contains(TEXT("Bridge"), ESearchCase::IgnoreCase) ||
		 Actor->GetClass()->GetName().Contains(TEXT("CityBase"), ESearchCase::IgnoreCase));
}
} // namespace

bool AeroGroundPlacement::TryProjectWorldPointToGround(
	UWorld* World,
	const FVector& WorldCm,
	FVector& OutProjectedWorldCm,
	FVector* OutSurfaceNormalWorld,
	const AActor* IgnoredActor)
{
	if (World == nullptr)
	{
		return false;
	}

	const FVector TraceStart = WorldCm + FVector(0.0f, 0.0f, GroundProjectionTraceHalfHeightCm);
	const FVector TraceEnd = WorldCm - FVector(0.0f, 0.0f, GroundProjectionTraceHalfHeightCm);
	TArray<FHitResult> HitResults;
	FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(AeroGroundPlacement), true);
	if (IgnoredActor != nullptr)
	{
		QueryParams.AddIgnoredActor(IgnoredActor);
	}

	if (!World->LineTraceMultiByChannel(HitResults, TraceStart, TraceEnd, ECC_Visibility, QueryParams))
	{
		return false;
	}

	const FHitResult* SelectedHit = nullptr;
	for (const FHitResult& HitResult : HitResults)
	{
		if (!HitResult.bBlockingHit)
		{
			continue;
		}

		if (SelectedHit == nullptr || HitResult.ImpactPoint.Z > SelectedHit->ImpactPoint.Z)
		{
			SelectedHit = &HitResult;
		}

		if (IsPreferredGroundActor(HitResult.GetActor()))
		{
			SelectedHit = &HitResult;
			break;
		}
	}

	if (SelectedHit == nullptr)
	{
		return false;
	}

	OutProjectedWorldCm = WorldCm;
	OutProjectedWorldCm.Z = SelectedHit->ImpactPoint.Z;
	if (OutSurfaceNormalWorld != nullptr)
	{
		*OutSurfaceNormalWorld = SelectedHit->ImpactNormal.GetSafeNormal();
		if (OutSurfaceNormalWorld->IsNearlyZero())
		{
			*OutSurfaceNormalWorld = FVector::UpVector;
		}
	}
	return true;
}

bool AeroGroundPlacement::ResolveGroundPlacement(
	UWorld* World,
	const FVector& RequestedWorldCm,
	FResolvedGroundPlacement& OutPlacement,
	const AActor* IgnoredActor)
{
	OutPlacement = FResolvedGroundPlacement();
	OutPlacement.RequestedWorldCm = RequestedWorldCm;
	OutPlacement.GroundWorldCm = RequestedWorldCm;
	OutPlacement.SurfaceNormalWorld = FVector::UpVector;
	if (World == nullptr)
	{
		return false;
	}

	bool bResolvedGround = false;
	if (UAeroPedNavSemanticSubsystem* PedNavSubsystem = World->GetSubsystem<UAeroPedNavSemanticSubsystem>())
	{
		FString AnchorId;
		if (PedNavSubsystem->ProjectWorldPointToGroundDetailed(
				RequestedWorldCm,
				OutPlacement.GroundWorldCm,
				OutPlacement.SurfaceNormalWorld,
				AnchorId))
		{
			OutPlacement.Source = TEXT("semantic");
			OutPlacement.bResolved = true;
			bResolvedGround = true;
		}
	}

	const FVector TraceAnchorWorldCm = bResolvedGround ? OutPlacement.GroundWorldCm : RequestedWorldCm;
	FVector TraceProjectedWorldCm = TraceAnchorWorldCm;
	FVector TraceSurfaceNormalWorld = FVector::UpVector;
	if (TryProjectWorldPointToGround(World, TraceAnchorWorldCm, TraceProjectedWorldCm, &TraceSurfaceNormalWorld, IgnoredActor))
	{
		if (!bResolvedGround || TraceProjectedWorldCm.Z > OutPlacement.GroundWorldCm.Z + GroundTraceLiftPreferenceCm)
		{
			OutPlacement.GroundWorldCm = TraceProjectedWorldCm;
			OutPlacement.SurfaceNormalWorld = TraceSurfaceNormalWorld;
			OutPlacement.Source = TEXT("trace");
		}
		OutPlacement.bResolved = true;
	}

	return OutPlacement.bResolved;
}
