#pragma once

#include "CoreMinimal.h"

class AActor;
class UWorld;

namespace AeroGroundPlacement
{
struct FResolvedGroundPlacement
{
	FVector RequestedWorldCm = FVector::ZeroVector;
	FVector GroundWorldCm = FVector::ZeroVector;
	FVector SurfaceNormalWorld = FVector::UpVector;
	FString Source;
	bool bResolved = false;
};

PEDESTRIANRUNTIME_API bool TryProjectWorldPointToGround(
	UWorld* World,
	const FVector& WorldCm,
	FVector& OutProjectedWorldCm,
	FVector* OutSurfaceNormalWorld = nullptr,
	const AActor* IgnoredActor = nullptr);

PEDESTRIANRUNTIME_API bool ResolveGroundPlacement(
	UWorld* World,
	const FVector& RequestedWorldCm,
	FResolvedGroundPlacement& OutPlacement,
	const AActor* IgnoredActor = nullptr);
} // namespace AeroGroundPlacement
