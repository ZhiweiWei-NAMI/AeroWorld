#include "HAL/IConsoleManager.h"

#include "Engine/World.h"
#include "PedestrianRuntimeLog.h"
#include "PedestrianWorldSubsystem.h"

namespace
{
bool ParseFloatToken(const FString& Token, float& OutValue)
{
	return LexTryParseString(OutValue, *Token);
}

bool ParseVector3(const TArray<FString>& Args, int32 StartIndex, FVector& OutVector)
{
	float X = 0.0f;
	float Y = 0.0f;
	float Z = 0.0f;
	if (!ParseFloatToken(Args[StartIndex + 0], X))
	{
		return false;
	}
	if (!ParseFloatToken(Args[StartIndex + 1], Y))
	{
		return false;
	}
	if (!ParseFloatToken(Args[StartIndex + 2], Z))
	{
		return false;
	}
	OutVector = FVector(X, Y, Z);
	return true;
}

bool EnsureUsage(const TArray<FString>& Args, int32 RequiredArgCount, const TCHAR* Usage, FOutputDevice& Ar)
{
	if (Args.Num() >= RequiredArgCount)
	{
		return true;
	}

	Ar.Logf(TEXT("usage: %s"), Usage);
	return false;
}

UPedestrianWorldSubsystem* ResolveSubsystem(UWorld* World, FOutputDevice& Ar)
{
	if (World == nullptr)
	{
		Ar.Log(TEXT("No world context available."));
		return nullptr;
	}

	if (!World->IsGameWorld())
	{
		Ar.Log(TEXT("Pedestrian commands require PIE or game world."));
		return nullptr;
	}

	UPedestrianWorldSubsystem* Subsystem = World->GetSubsystem<UPedestrianWorldSubsystem>();
	if (Subsystem == nullptr)
	{
		Ar.Log(TEXT("PedestrianWorldSubsystem is unavailable."));
	}
	return Subsystem;
}

static FAutoConsoleCommand GPedResetCmd(
	TEXT("ped.reset"),
	TEXT("ped.reset <PedId> <X> <Y> <Z> <Yaw>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.reset <PedId> <X> <Y> <Z> <Yaw>");
			if (!EnsureUsage(Args, 5, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			FVector Location = FVector::ZeroVector;
			float Yaw = 0.0f;
			if (!ParseVector3(Args, 1, Location) || !ParseFloatToken(Args[4], Yaw))
			{
				Ar.Logf(TEXT("usage: %s"), Usage);
				return;
			}

			if (!Subsystem->ExecReset(Args[0], Location, Yaw))
			{
				Ar.Logf(TEXT("ped.reset failed: PedId='%s'"), *Args[0]);
			}
		}));

static FAutoConsoleCommand GPedObserveCmd(
	TEXT("ped.observe"),
	TEXT("ped.observe <PedId>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.observe <PedId>");
			if (!EnsureUsage(Args, 1, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			if (!Subsystem->ExecObserve(Args[0]))
			{
				Ar.Logf(TEXT("ped.observe failed: PedId='%s'"), *Args[0]);
			}
		}));

static FAutoConsoleCommand GPedCommitCrossCmd(
	TEXT("ped.commit_cross"),
	TEXT("ped.commit_cross <PedId> <X> <Y> <Z> <SpeedCmPerSec>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.commit_cross <PedId> <X> <Y> <Z> <SpeedCmPerSec>");
			if (!EnsureUsage(Args, 5, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			FVector Target = FVector::ZeroVector;
			float SpeedCmPerSec = 0.0f;
			if (!ParseVector3(Args, 1, Target) || !ParseFloatToken(Args[4], SpeedCmPerSec))
			{
				Ar.Logf(TEXT("usage: %s"), Usage);
				return;
			}

			if (!Subsystem->ExecCommitCross(Args[0], Target, SpeedCmPerSec))
			{
				Ar.Logf(TEXT("ped.commit_cross failed: PedId='%s'"), *Args[0]);
			}
		}));

static FAutoConsoleCommand GPedStopCmd(
	TEXT("ped.stop"),
	TEXT("ped.stop <PedId>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.stop <PedId>");
			if (!EnsureUsage(Args, 1, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			if (!Subsystem->ExecStop(Args[0]))
			{
				Ar.Logf(TEXT("ped.stop failed: PedId='%s'"), *Args[0]);
			}
		}));

static FAutoConsoleCommand GPedSetTargetCmd(
	TEXT("ped.set_target"),
	TEXT("ped.set_target <PedId> <X> <Y> <Z> <SpeedCmPerSec>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.set_target <PedId> <X> <Y> <Z> <SpeedCmPerSec>");
			if (!EnsureUsage(Args, 5, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			FVector Target = FVector::ZeroVector;
			float SpeedCmPerSec = 0.0f;
			if (!ParseVector3(Args, 1, Target) || !ParseFloatToken(Args[4], SpeedCmPerSec))
			{
				Ar.Logf(TEXT("usage: %s"), Usage);
				return;
			}

			if (!Subsystem->ExecSetTarget(Args[0], Target, SpeedCmPerSec))
			{
				Ar.Logf(TEXT("ped.set_target failed: PedId='%s'"), *Args[0]);
			}
		}));

static FAutoConsoleCommand GPedSetVariantCmd(
	TEXT("ped.set_variant"),
	TEXT("ped.set_variant <PedId> <VariantId>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.set_variant <PedId> <VariantId>");
			if (!EnsureUsage(Args, 2, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			const FString VariantText = Args[1].TrimStartAndEnd();
			if (VariantText.IsEmpty())
			{
				Ar.Logf(TEXT("usage: %s"), Usage);
				return;
			}

			if (!Subsystem->ExecSetVariant(Args[0], FName(*VariantText)))
			{
				Ar.Logf(TEXT("ped.set_variant failed: PedId='%s' VariantId='%s'"), *Args[0], *VariantText);
			}
		}));

static FAutoConsoleCommand GPedSpawnCmd(
	TEXT("ped.spawn"),
	TEXT("ped.spawn <PedId> <X> <Y> <Z> <Yaw> [VariantId]"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.spawn <PedId> <X> <Y> <Z> <Yaw> [VariantId]");
			if (!EnsureUsage(Args, 5, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			FVector Location = FVector::ZeroVector;
			float Yaw = 0.0f;
			if (!ParseVector3(Args, 1, Location) || !ParseFloatToken(Args[4], Yaw))
			{
				Ar.Logf(TEXT("usage: %s"), Usage);
				return;
			}

			FName VariantId = NAME_None;
			if (Args.Num() >= 6)
			{
				const FString VariantText = Args[5].TrimStartAndEnd();
				if (!VariantText.IsEmpty())
				{
					VariantId = FName(*VariantText);
				}
			}

			if (!Subsystem->ExecSpawn(Args[0], Location, Yaw, VariantId))
			{
				Ar.Logf(TEXT("ped.spawn failed: PedId='%s'"), *Args[0]);
				return;
			}

			if (VariantId.IsNone())
			{
				Ar.Logf(TEXT("ped.spawn ok: PedId='%s'"), *Args[0]);
			}
			else
			{
				Ar.Logf(TEXT("ped.spawn ok: PedId='%s' VariantId='%s'"), *Args[0], *VariantId.ToString());
			}
		}));

static FAutoConsoleCommand GPedReleaseCmd(
	TEXT("ped.release"),
	TEXT("ped.release <PedId>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			static const TCHAR* Usage = TEXT("ped.release <PedId>");
			if (!EnsureUsage(Args, 1, Usage, Ar))
			{
				return;
			}

			UPedestrianWorldSubsystem* Subsystem = ResolveSubsystem(World, Ar);
			if (Subsystem == nullptr)
			{
				return;
			}

			if (!Subsystem->ExecRelease(Args[0]))
			{
				Ar.Logf(TEXT("ped.release failed: PedId='%s'"), *Args[0]);
				return;
			}

			Ar.Logf(TEXT("ped.release ok: PedId='%s'"), *Args[0]);
		}));
}
