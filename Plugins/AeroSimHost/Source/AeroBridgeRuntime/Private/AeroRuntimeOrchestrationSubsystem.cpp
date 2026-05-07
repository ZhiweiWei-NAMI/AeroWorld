#include "AeroRuntimeOrchestrationSubsystem.h"

#include "Animation/AnimationAsset.h"
#include "Async/Async.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "HAL/PlatformProcess.h"
#include "PedestrianCharacter.h"
#include "PedestrianWorldSubsystem.h"
#include "SimMode/SimModeBase.h"
#include "UObject/SoftObjectPath.h"
#include "Vehicles/Multirotor/SimModeWorldMultiRotor.h"
#include "vehicles/multirotor/api/MultirotorApiBase.hpp"

#include <exception>

DEFINE_LOG_CATEGORY_STATIC(LogAeroRuntimeOrchestration, Log, All);

namespace
{
constexpr TCHAR* RuntimeValidationUavPawnPath = TEXT("Class'/AeroWorldContent/Blueprints/UAV/BP_AW_UAV_Inspection_Quad_01.BP_AW_UAV_Inspection_Quad_01_C'");
constexpr TCHAR* RuntimeValidationUavVehicleType = TEXT("simpleflight");
constexpr float DefaultMultirotorMoveTimeoutSec = 20.0f;
constexpr float DefaultMultirotorReadyTimeoutSec = 3.0f;
constexpr float DefaultMultirotorTakeoffTimeoutSec = 5.0f;
constexpr float DefaultMultirotorSettleTimeoutSec = 2.0f;
constexpr float DefaultMultirotorPollIntervalSec = 0.1f;
constexpr float DefaultMoveStartSpeedMps = 0.75f;
constexpr float DefaultMoveVerticalToleranceM = 0.35f;
constexpr float DefaultMoveHorizontalToleranceM = 0.75f;
constexpr float DefaultMoveMinDurationSec = 0.25f;
constexpr float DefaultAdaptiveLookahead = 1.0f;
constexpr float DefaultLookahead = -1.0f;

FString ToRuntimeMoveStateLogString(const EAeroRuntimeMoveState State)
{
	switch (State)
	{
	case EAeroRuntimeMoveState::Running:
		return TEXT("Running");
	case EAeroRuntimeMoveState::Succeeded:
		return TEXT("Succeeded");
	case EAeroRuntimeMoveState::Failed:
		return TEXT("Failed");
	case EAeroRuntimeMoveState::Cancelled:
		return TEXT("Cancelled");
	default:
		return TEXT("Idle");
	}
}

FString ToLandedStateLogString(const msr::airlib::LandedState State)
{
	switch (State)
	{
	case msr::airlib::LandedState::Flying:
		return TEXT("Flying");
	case msr::airlib::LandedState::Landed:
	default:
		return TEXT("Landed");
	}
}

msr::airlib::Pose MakeGlobalNedPose(ASimModeBase* SimModeActor, const FVector& WorldLocationCm, const FRotator& WorldRotation)
{
	return SimModeActor->getGlobalNedTransform().toGlobalNed(FTransform(WorldRotation, WorldLocationCm));
}
} // namespace

struct UAeroRuntimeOrchestrationSubsystem::FMultirotorMoveJobState
{
	mutable FCriticalSection Mutex;
	EAeroRuntimeMoveState State = EAeroRuntimeMoveState::Idle;
	FString Message;
	FVector TargetWorldCm = FVector::ZeroVector;
	float VelocityMps = 0.0f;
	uint64 CommandId = 0;
	TAtomic<bool> bCancelRequested { false };
};

bool UAeroRuntimeOrchestrationSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroRuntimeOrchestrationSubsystem::Deinitialize()
{
	TArray<FString> VehicleNames;
	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		MultirotorMoveJobs.GetKeys(VehicleNames);
	}

	FString Error;
	for (const FString& VehicleName : VehicleNames)
	{
		CancelTrackedMove(VehicleName, true, Error);
	}

	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		MultirotorMoveJobs.Reset();
	}
	RuntimeMultirotorSpawnWorldCm.Empty();

	Super::Deinitialize();
}

bool UAeroRuntimeOrchestrationSubsystem::GetCurrentAirSimCapabilities(FAeroRuntimeAirSimCapabilities& OutCapabilities) const
{
	OutCapabilities = FAeroRuntimeAirSimCapabilities();

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		return false;
	}

	OutCapabilities.SimModeName = SimModeActor->GetClass()->GetName();
	OutCapabilities.bSupportsMultirotors = Cast<ASimModeWorldMultiRotor>(SimModeActor) != nullptr;
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SpawnPedestrian(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	const FName VariantId,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSpawn(PedId, WorldLocationCm, YawDeg, VariantId, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to spawn pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ResetPedestrian(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecReset(PedId, WorldLocationCm, YawDeg, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to reset pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianFramePose(
	const FString& PedId,
	const FVector& WorldLocationCm,
	const float YawDeg,
	const bool bWalking,
	const float SpeedCmPerSec,
	FString& OutError,
	const bool bUseProvidedGroundPoint) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetFramePose(PedId, WorldLocationCm, YawDeg, bWalking, SpeedCmPerSec, bUseProvidedGroundPoint))
	{
		OutError = FString::Printf(TEXT("Failed to set frame pose for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianTarget(
	const FString& PedId,
	const FVector& TargetWorldCm,
	const float SpeedCmPerSec,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetTarget(PedId, TargetWorldCm, SpeedCmPerSec))
	{
		OutError = FString::Printf(TEXT("Failed to set target for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ObservePedestrian(const FString& PedId, const FName StartSection, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (StartSection.IsNone())
	{
		if (!PedSubsystem->ExecObserve(PedId))
		{
			OutError = FString::Printf(TEXT("Failed to observe pedestrian '%s'."), *PedId);
			return false;
		}
		return true;
	}

	APedestrianCharacter* Ped = PedSubsystem->FindPedestrian(PedId);
	if (Ped == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to observe pedestrian '%s'."), *PedId);
		return false;
	}

	Ped->CmdPlayObserve(StartSection);
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::PlayPedestrianAnimation(
	const FString& PedId,
	const FString& AnimationAssetPath,
	const FName StartSection,
	const float PlayRate,
	const int32 LoopCount,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	const FString TrimmedAnimationAssetPath = AnimationAssetPath.TrimStartAndEnd();
	if (TrimmedAnimationAssetPath.IsEmpty())
	{
		OutError = TEXT("Animation asset path is required.");
		return false;
	}

	APedestrianCharacter* Ped = PedSubsystem->FindPedestrian(PedId);
	if (Ped == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to find pedestrian '%s'."), *PedId);
		return false;
	}

	UAnimationAsset* AnimationAsset = Cast<UAnimationAsset>(FSoftObjectPath(TrimmedAnimationAssetPath).TryLoad());
	if (AnimationAsset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load animation asset '%s'."), *TrimmedAnimationAssetPath);
		return false;
	}

	if (!Ped->CmdPlayAnimationAsset(AnimationAsset, StartSection, PlayRate, LoopCount))
	{
		OutError = FString::Printf(TEXT("Failed to play animation '%s' for pedestrian '%s'."), *TrimmedAnimationAssetPath, *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::CommitPedestrianCross(
	const FString& PedId,
	const FVector& TargetWorldCm,
	const float SpeedCmPerSec,
	FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecCommitCross(PedId, TargetWorldCm, SpeedCmPerSec))
	{
		OutError = FString::Printf(TEXT("Failed to commit cross for pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::StopPedestrian(const FString& PedId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecStop(PedId))
	{
		OutError = FString::Printf(TEXT("Failed to stop pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SetPedestrianVariant(const FString& PedId, const FName VariantId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecSetVariant(PedId, VariantId))
	{
		OutError = FString::Printf(TEXT("Failed to set variant '%s' for pedestrian '%s'."), *VariantId.ToString(), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ReleasePedestrian(const FString& PedId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ExecRelease(PedId))
	{
		OutError = FString::Printf(TEXT("Failed to release pedestrian '%s'."), *PedId);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::SpawnCrowd(
	const FCrowdSpawnRequest& Request,
	FCrowdSpawnResult& OutResult,
	FString& OutError) const
{
	OutResult = FCrowdSpawnResult();

	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	OutResult = PedSubsystem->SpawnCrowd(Request);
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ClearCrowdGroup(const FName GroupId, FString& OutError) const
{
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	if (!PedSubsystem->ClearCrowdGroup(GroupId))
	{
		OutError = FString::Printf(TEXT("Failed to clear crowd group '%s'."), *GroupId.ToString());
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::RespawnCrowd(
	const FName GroupId,
	const int32 NewSeed,
	FCrowdSpawnResult& OutResult,
	FString& OutError) const
{
	OutResult = FCrowdSpawnResult();

	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(OutError);
	if (PedSubsystem == nullptr)
	{
		return false;
	}

	OutResult = PedSubsystem->RespawnCrowd(GroupId, NewSeed);
	return true;
}

AActor* UAeroRuntimeOrchestrationSubsystem::ResolvePedestrianActor(const FString& PedId) const
{
	FString Error;
	UPedestrianWorldSubsystem* PedSubsystem = ResolvePedestrianSubsystem(Error);
	return PedSubsystem != nullptr ? PedSubsystem->FindPedestrian(PedId) : nullptr;
}

bool UAeroRuntimeOrchestrationSubsystem::CreateRuntimeMultirotor(
	const FString& VehicleName,
	const FVector& WorldLocationCm,
	const FRotator& WorldRotation,
	FString& OutError)
{
	OutError.Reset();

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable.");
		return false;
	}

	FAeroRuntimeAirSimCapabilities Capabilities;
	GetCurrentAirSimCapabilities(Capabilities);
	if (!Capabilities.bSupportsMultirotors)
	{
		OutError = FString::Printf(TEXT("Current AirSim SimMode '%s' does not support runtime multirotors."), *Capabilities.SimModeName);
		return false;
	}

	if (ResolveSpawnedPawnByVehicleName(VehicleName) != nullptr)
	{
		FString RemoveError;
		if (!RemoveRuntimeVehicle(VehicleName, RemoveError))
		{
			OutError = RemoveError;
			return false;
		}
	}

	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		MultirotorMoveJobs.Remove(VehicleName);
	}

	UE_LOG(
		LogAeroRuntimeOrchestration,
		Log,
		TEXT("CreateRuntimeMultirotor request: name='%s' type='%s' pawn_path='%s' location='%s' rotation='%s'."),
		*VehicleName,
		RuntimeValidationUavVehicleType,
		RuntimeValidationUavPawnPath,
		*WorldLocationCm.ToString(),
		*WorldRotation.ToString());

	const msr::airlib::Pose SpawnPose = MakeGlobalNedPose(SimModeActor, WorldLocationCm, WorldRotation);
	const bool bCreated = SimModeActor->createVehicleAtRuntime(
		TCHAR_TO_UTF8(*VehicleName),
		TCHAR_TO_UTF8(RuntimeValidationUavVehicleType),
		SpawnPose,
		TCHAR_TO_UTF8(RuntimeValidationUavPawnPath));
	if (!bCreated)
	{
		OutError = FString::Printf(TEXT("AirSim runtime UAV creation failed: name='%s'."), *VehicleName);
		return false;
	}

	if (ResolveSpawnedPawnByVehicleName(VehicleName) == nullptr)
	{
		OutError = FString::Printf(TEXT("AirSim created runtime UAV '%s' but no pawn was resolved."), *VehicleName);
		return false;
	}

	RuntimeMultirotorSpawnWorldCm.Add(VehicleName, WorldLocationCm);
	UE_LOG(LogAeroRuntimeOrchestration, Log, TEXT("CreateRuntimeMultirotor succeeded: name='%s'."), *VehicleName);
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::EnableApiControl(const FString& VehicleName, const bool bEnable, FString& OutError) const
{
	OutError.Reset();

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable.");
		return false;
	}

	msr::airlib::VehicleApiBase* VehicleApi = SimModeActor->getApiProvider()->getVehicleApi(TCHAR_TO_UTF8(*VehicleName));
	if (VehicleApi == nullptr)
	{
		OutError = FString::Printf(TEXT("AirSim vehicle API is unavailable for '%s'."), *VehicleName);
		return false;
	}

	VehicleApi->enableApiControl(bEnable);
	if (bEnable && !VehicleApi->isApiControlEnabled())
	{
		OutError = FString::Printf(TEXT("Failed to enable API control for '%s'."), *VehicleName);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::ArmIfSupported(const FString& VehicleName, FString& OutError) const
{
	OutError.Reset();

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable.");
		return false;
	}

	msr::airlib::VehicleApiBase* VehicleApi = SimModeActor->getApiProvider()->getVehicleApi(TCHAR_TO_UTF8(*VehicleName));
	if (VehicleApi == nullptr)
	{
		OutError = FString::Printf(TEXT("AirSim vehicle API is unavailable for '%s'."), *VehicleName);
		return false;
	}

	if (!VehicleApi->canArm())
	{
		OutError = FString::Printf(TEXT("AirSim vehicle '%s' cannot arm in the current state."), *VehicleName);
		return false;
	}

	if (!VehicleApi->armDisarm(true))
	{
		OutError = FString::Printf(TEXT("Failed to arm AirSim vehicle '%s'."), *VehicleName);
		return false;
	}

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::MoveMultirotorToPosition(
	const FString& VehicleName,
	const FVector& TargetWorldCm,
	const float VelocityMps,
	FString& OutError)
{
	OutError.Reset();

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable.");
		return false;
	}

	FAeroRuntimeAirSimCapabilities Capabilities;
	GetCurrentAirSimCapabilities(Capabilities);
	if (!Capabilities.bSupportsMultirotors)
	{
		OutError = FString::Printf(TEXT("Current AirSim SimMode '%s' does not support multirotor moveTo."), *Capabilities.SimModeName);
		return false;
	}

	FString CancelError;
	if (!CancelTrackedMove(VehicleName, true, CancelError))
	{
		OutError = CancelError;
		return false;
	}

	msr::airlib::VehicleApiBase* VehicleApiBase = SimModeActor->getApiProvider()->getVehicleApi(TCHAR_TO_UTF8(*VehicleName));
	if (VehicleApiBase == nullptr)
	{
		OutError = FString::Printf(TEXT("AirSim multirotor API is unavailable for '%s'."), *VehicleName);
		return false;
	}
	auto* MultirotorApi = static_cast<msr::airlib::MultirotorApiBase*>(VehicleApiBase);

	const FVector* SpawnWorldCm = RuntimeMultirotorSpawnWorldCm.Find(VehicleName);
	FVector LocalOriginWorldCm = SpawnWorldCm != nullptr ? *SpawnWorldCm : FVector::ZeroVector;
	if (SpawnWorldCm == nullptr)
	{
		APawn* RuntimePawn = ResolveSpawnedPawnByVehicleName(VehicleName);
		if (!IsValid(RuntimePawn))
		{
			OutError = FString::Printf(TEXT("AirSim runtime pawn is unavailable for '%s'."), *VehicleName);
			return false;
		}
		LocalOriginWorldCm = RuntimePawn->GetActorLocation();
		RuntimeMultirotorSpawnWorldCm.Add(VehicleName, LocalOriginWorldCm);
		UE_LOG(
			LogAeroRuntimeOrchestration,
			Warning,
			TEXT("MoveMultirotorToPosition rebuilt missing spawn origin from current pawn location for '%s': origin='%s'."),
			*VehicleName,
			*LocalOriginWorldCm.ToString());
	}
	const msr::airlib::Vector3r TargetLocalNed =
		SimModeActor->getGlobalNedTransform().toGlobalNed(TargetWorldCm) -
		SimModeActor->getGlobalNedTransform().toGlobalNed(LocalOriginWorldCm);
	const uint64 CommandId = NextMultirotorMoveCommandId++;

	TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe> Job = MakeShared<FMultirotorMoveJobState, ESPMode::ThreadSafe>();
	Job->State = EAeroRuntimeMoveState::Running;
	Job->Message = FString::Printf(TEXT("Move started toward '%s'."), *TargetWorldCm.ToString());
	Job->TargetWorldCm = TargetWorldCm;
	Job->VelocityMps = VelocityMps;
	Job->CommandId = CommandId;

	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		MultirotorMoveJobs.Add(VehicleName, Job);
	}

	UE_LOG(
		LogAeroRuntimeOrchestration,
		Log,
		TEXT("MoveMultirotorToPosition async start: vehicle='%s' command_id=%llu target='%s' target_local_ned='(%.3f, %.3f, %.3f)' velocity=%.2f."),
		*VehicleName,
		CommandId,
		*TargetWorldCm.ToString(),
		static_cast<double>(TargetLocalNed.x()),
		static_cast<double>(TargetLocalNed.y()),
		static_cast<double>(TargetLocalNed.z()),
		VelocityMps);

	Async(EAsyncExecution::ThreadPool, [Job, VehicleApiBase, MultirotorApi, VehicleName, TargetWorldCm, VelocityMps, TargetLocalNed]() {
		bool bMoveCompleted = false;
		FString ResultMessage = TEXT("AirSim multirotor staged move failed.");

		try
		{
			msr::airlib::MultirotorState LastState;
			bool bHasState = false;
			const auto SetRunningMessage = [&](const FString& Message) {
				FScopeLock Lock(&Job->Mutex);
				if (Job->State == EAeroRuntimeMoveState::Running)
				{
					Job->Message = Message;
				}
			};
			const auto UpdateState = [&]() {
				LastState = MultirotorApi->getMultirotorState();
				bHasState = true;
			};
			const auto GetLinearSpeedMps = [&]() -> float {
				const auto& Linear = LastState.kinematics_estimated.twist.linear;
				return FVector(
						   static_cast<float>(Linear.x()),
						   static_cast<float>(Linear.y()),
						   static_cast<float>(Linear.z()))
					.Size();
			};
			const auto GetCurrentNed = [&]() -> msr::airlib::Vector3r {
				return LastState.kinematics_estimated.pose.position;
			};
			const auto IsCancelled = [&](const TCHAR* CancelMessage) -> bool {
				if (!Job->bCancelRequested.Load())
				{
					return false;
				}

				ResultMessage = CancelMessage;
				return true;
			};
			const auto SleepWithCancellation = [&](const float DurationSec, const TCHAR* CancelMessage) -> bool {
				const double SleepUntil = FPlatformTime::Seconds() + DurationSec;
				while (FPlatformTime::Seconds() < SleepUntil)
				{
					if (IsCancelled(CancelMessage))
					{
						return false;
					}
					FPlatformProcess::Sleep(FMath::Min(DefaultMultirotorPollIntervalSec, static_cast<float>(SleepUntil - FPlatformTime::Seconds())));
				}
				return true;
			};
			const auto WaitForReady = [&](const float TimeoutSec) -> bool {
				const double Deadline = FPlatformTime::Seconds() + TimeoutSec;
				while (FPlatformTime::Seconds() < Deadline)
				{
					if (IsCancelled(TEXT("Move cancelled before vehicle became ready.")))
					{
						return false;
					}

					UpdateState();
					if (LastState.ready)
					{
						return true;
					}

					FPlatformProcess::Sleep(DefaultMultirotorPollIntervalSec);
				}

				if (!bHasState)
				{
					ResultMessage = FString::Printf(TEXT("Failed to read AirSim multirotor state for '%s'."), *VehicleName);
				}
				else
				{
					ResultMessage = FString::Printf(
						TEXT("AirSim multirotor '%s' was not ready before move: landed='%s' can_arm=%s ready_message='%s'."),
						*VehicleName,
						*ToLandedStateLogString(LastState.landed_state),
						LastState.can_arm ? TEXT("true") : TEXT("false"),
						UTF8_TO_TCHAR(LastState.ready_message.c_str()));
				}
				return false;
			};
			const auto WaitForStableFlight = [&](const float TimeoutSec, const TCHAR* CancelMessage) -> bool {
				const double Deadline = FPlatformTime::Seconds() + TimeoutSec;
				while (FPlatformTime::Seconds() < Deadline)
				{
					if (IsCancelled(CancelMessage))
					{
						return false;
					}

					UpdateState();
					if (LastState.ready && LastState.landed_state == msr::airlib::LandedState::Flying && GetLinearSpeedMps() <= DefaultMoveStartSpeedMps)
					{
						return true;
					}

					FPlatformProcess::Sleep(DefaultMultirotorPollIntervalSec);
				}
				return false;
			};
			const auto LogStage = [&](const TCHAR* Stage, const FString& Message) {
				SetRunningMessage(Message);
				UE_LOG(
					LogAeroRuntimeOrchestration,
					Log,
					TEXT("MoveMultirotorToPosition stage: vehicle='%s' command_id=%llu stage='%s' message='%s'."),
					*VehicleName,
					Job->CommandId,
					Stage,
					*Message);
			};

			const float EffectiveVelocityMps = FMath::Max(0.5f, VelocityMps);
			const msr::airlib::YawMode HoldYaw(false, 0.0f);

			if (!WaitForReady(DefaultMultirotorReadyTimeoutSec))
			{
				bMoveCompleted = false;
			}
			else
			{
				UpdateState();
				UE_LOG(
					LogAeroRuntimeOrchestration,
					Log,
					TEXT("MoveMultirotorToPosition preflight: vehicle='%s' landed='%s' speed_mps=%.2f can_arm=%s ready_message='%s'."),
					*VehicleName,
					*ToLandedStateLogString(LastState.landed_state),
					GetLinearSpeedMps(),
					LastState.can_arm ? TEXT("true") : TEXT("false"),
					UTF8_TO_TCHAR(LastState.ready_message.c_str()));

				if (!VehicleApiBase->isApiControlEnabled())
				{
					LogStage(TEXT("api_control"), TEXT("Enabling API control."));
					VehicleApiBase->enableApiControl(true);
					if (!VehicleApiBase->isApiControlEnabled())
					{
						ResultMessage = FString::Printf(TEXT("Failed to enable API control for AirSim multirotor '%s'."), *VehicleName);
					}
				}

				if (ResultMessage == TEXT("AirSim multirotor staged move failed.") && LastState.landed_state == msr::airlib::LandedState::Landed)
				{
					if (!LastState.can_arm)
					{
						ResultMessage = FString::Printf(TEXT("AirSim multirotor '%s' cannot arm before staged move."), *VehicleName);
					}
					else
					{
						LogStage(TEXT("arm"), TEXT("Arming multirotor."));
						if (!VehicleApiBase->armDisarm(true))
						{
							ResultMessage = FString::Printf(TEXT("Failed to arm AirSim multirotor '%s' before staged move."), *VehicleName);
						}
					}
				}

				if (ResultMessage != TEXT("AirSim multirotor staged move failed."))
				{
					bMoveCompleted = false;
				}
				else
				{
					if (LastState.landed_state == msr::airlib::LandedState::Landed)
					{
						const float PreTakeoffSpeedMps = GetLinearSpeedMps();
						if (PreTakeoffSpeedMps > DefaultMoveStartSpeedMps)
						{
							UE_LOG(
								LogAeroRuntimeOrchestration,
								Warning,
								TEXT("MoveMultirotorToPosition saw '%s' as Landed but already moving at %.2f m/s; skipping takeoff and continuing with runtime move."),
								*VehicleName,
								PreTakeoffSpeedMps);
						}
						else
						{
							LogStage(TEXT("takeoff"), TEXT("Executing takeoff before local-NED runtime move."));
							if (!MultirotorApi->takeoff(DefaultMultirotorTakeoffTimeoutSec))
							{
								if (Job->bCancelRequested.Load())
								{
									ResultMessage = TEXT("Move cancelled during takeoff.");
								}
								else
								{
									UpdateState();
									if (LastState.ready && LastState.landed_state == msr::airlib::LandedState::Flying)
									{
										UE_LOG(
											LogAeroRuntimeOrchestration,
											Warning,
											TEXT("MoveMultirotorToPosition takeoff returned false for '%s', but state is Flying; continuing with runtime move."),
											*VehicleName);
									}
									else
									{
										ResultMessage = FString::Printf(TEXT("AirSim takeoff failed before runtime move for '%s'."), *VehicleName);
									}
								}
							}
							else if (!SleepWithCancellation(0.25f, TEXT("Move cancelled after takeoff.")))
							{
								bMoveCompleted = false;
							}
						}
					}
				}

				if (ResultMessage != TEXT("AirSim multirotor staged move failed."))
				{
					bMoveCompleted = false;
				}
				else
				{
					LogStage(TEXT("hover"), TEXT("Stabilizing before local-NED runtime move."));
					const bool bHoverAccepted = MultirotorApi->hover();
					if (!bHoverAccepted)
					{
						UE_LOG(
							LogAeroRuntimeOrchestration,
							Warning,
							TEXT("MoveMultirotorToPosition hover stabilization was rejected for '%s'; continuing."),
							*VehicleName);
					}

					WaitForStableFlight(DefaultMultirotorSettleTimeoutSec, TEXT("Move cancelled while stabilizing before translation."));
					UpdateState();
					msr::airlib::Vector3r CurrentNed = GetCurrentNed();
					float RemainingXYM = FMath::Sqrt(
						FMath::Square(static_cast<float>(TargetLocalNed.x() - CurrentNed.x())) +
						FMath::Square(static_cast<float>(TargetLocalNed.y() - CurrentNed.y())));
					float RemainingZM = FMath::Abs(static_cast<float>(TargetLocalNed.z() - CurrentNed.z()));
					UE_LOG(
						LogAeroRuntimeOrchestration,
						Log,
						TEXT("MoveMultirotorToPosition local target check: vehicle='%s' landed='%s' speed_mps=%.2f current_local_ned='(%.3f, %.3f, %.3f)' target_local_ned='(%.3f, %.3f, %.3f)' target='%s'."),
						*VehicleName,
						*ToLandedStateLogString(LastState.landed_state),
						GetLinearSpeedMps(),
						static_cast<double>(CurrentNed.x()),
						static_cast<double>(CurrentNed.y()),
						static_cast<double>(CurrentNed.z()),
						static_cast<double>(TargetLocalNed.x()),
						static_cast<double>(TargetLocalNed.y()),
						static_cast<double>(TargetLocalNed.z()),
						*TargetWorldCm.ToString());

					if (RemainingXYM <= DefaultMoveHorizontalToleranceM && RemainingZM <= DefaultMoveVerticalToleranceM * 2.0f)
					{
						bMoveCompleted = true;
						ResultMessage = FString::Printf(
							TEXT("Runtime multirotor already within tolerance for '%s': residual_xy=%.2f m residual_z=%.2f m."),
							*TargetWorldCm.ToString(),
							RemainingXYM,
							RemainingZM);
					}
					else
					{
						LogStage(
							TEXT("move_to_position"),
							FString::Printf(
								TEXT("Moving to local NED target (%.2f, %.2f, %.2f) at %.2f m/s."),
								static_cast<double>(TargetLocalNed.x()),
								static_cast<double>(TargetLocalNed.y()),
								static_cast<double>(TargetLocalNed.z()),
								EffectiveVelocityMps));
						const bool bMoveAccepted = MultirotorApi->moveToPosition(
							TargetLocalNed.x(),
							TargetLocalNed.y(),
							TargetLocalNed.z(),
							EffectiveVelocityMps,
							DefaultMultirotorMoveTimeoutSec,
							msr::airlib::DrivetrainType::MaxDegreeOfFreedom,
							HoldYaw,
							DefaultLookahead,
							DefaultAdaptiveLookahead);
						if (!bMoveAccepted)
						{
							ResultMessage = Job->bCancelRequested.Load()
								? TEXT("Move cancelled during moveToPosition.")
								: FString::Printf(TEXT("moveToPosition returned false for local NED target '%s'."), *TargetWorldCm.ToString());
						}
						else
						{
							UpdateState();
							CurrentNed = GetCurrentNed();
							RemainingXYM = FMath::Sqrt(
								FMath::Square(static_cast<float>(TargetLocalNed.x() - CurrentNed.x())) +
								FMath::Square(static_cast<float>(TargetLocalNed.y() - CurrentNed.y())));
							RemainingZM = FMath::Abs(static_cast<float>(TargetLocalNed.z() - CurrentNed.z()));
							if (RemainingXYM <= DefaultMoveHorizontalToleranceM && RemainingZM <= DefaultMoveVerticalToleranceM * 2.0f)
							{
								bMoveCompleted = true;
								ResultMessage = FString::Printf(
									TEXT("Runtime multirotor move completed to '%s' with residual_xy=%.2f m residual_z=%.2f m."),
									*TargetWorldCm.ToString(),
									RemainingXYM,
									RemainingZM);
							}
							else
							{
								ResultMessage = FString::Printf(
									TEXT("Runtime multirotor move ended outside tolerance for '%s': residual_xy=%.2f m residual_z=%.2f m."),
									*TargetWorldCm.ToString(),
									RemainingXYM,
									RemainingZM);
							}
						}
					}
				}
			}
		}
		catch (const std::exception& Ex)
		{
			ResultMessage = FString::Printf(TEXT("staged move exception: %s"), UTF8_TO_TCHAR(Ex.what()));
		}
		catch (...)
		{
			ResultMessage = TEXT("staged move threw an unknown exception.");
		}

		EAeroRuntimeMoveState FinalState = EAeroRuntimeMoveState::Failed;
		FString FinalMessage;
		{
			FScopeLock Lock(&Job->Mutex);
			if (Job->bCancelRequested.Load())
			{
				Job->State = EAeroRuntimeMoveState::Cancelled;
				Job->Message = TEXT("Move cancelled.");
			}
			else if (bMoveCompleted)
			{
				Job->State = EAeroRuntimeMoveState::Succeeded;
				Job->Message = ResultMessage;
			}
			else
			{
				Job->State = EAeroRuntimeMoveState::Failed;
				Job->Message = ResultMessage;
			}
			FinalState = Job->State;
			FinalMessage = Job->Message;
		}

		UE_LOG(
			LogAeroRuntimeOrchestration,
			Log,
			TEXT("MoveMultirotorToPosition async finish: vehicle='%s' command_id=%llu state='%s' message='%s'."),
			*VehicleName,
			Job->CommandId,
			*ToRuntimeMoveStateLogString(FinalState),
			*FinalMessage);
	});

	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::GetMultirotorMoveStatus(
	const FString& VehicleName,
	FAeroRuntimeMoveStatus& OutStatus,
	FString& OutError) const
{
	OutStatus = FAeroRuntimeMoveStatus();
	OutError.Reset();

	TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe> Job;
	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		if (const TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe>* ExistingJob = MultirotorMoveJobs.Find(VehicleName))
		{
			Job = *ExistingJob;
		}
	}

	if (!Job.IsValid())
	{
		OutStatus.Message = TEXT("No multirotor move job recorded.");
		return true;
	}

	FScopeLock Lock(&Job->Mutex);
	OutStatus.State = Job->State;
	OutStatus.Message = Job->Message;
	OutStatus.TargetWorldCm = Job->TargetWorldCm;
	OutStatus.VelocityMps = Job->VelocityMps;
	return true;
}

bool UAeroRuntimeOrchestrationSubsystem::RemoveRuntimeVehicle(const FString& VehicleName, FString& OutError)
{
	OutError.Reset();

	if (!CancelTrackedMove(VehicleName, true, OutError))
	{
		return false;
	}

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable.");
		return false;
	}

	msr::airlib::VehicleApiBase* VehicleApi = SimModeActor->getApiProvider()->getVehicleApi(TCHAR_TO_UTF8(*VehicleName));
	if (VehicleApi != nullptr)
	{
		VehicleApi->cancelLastTask();
		VehicleApi->enableApiControl(false);
		VehicleApi->armDisarm(false);
	}

	if (!SimModeActor->removeVehicleAtRuntime(TCHAR_TO_UTF8(*VehicleName)))
	{
		OutError = FString::Printf(TEXT("Failed to remove AirSim runtime vehicle '%s'."), *VehicleName);
		return false;
	}

	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		MultirotorMoveJobs.Remove(VehicleName);
	}
	RuntimeMultirotorSpawnWorldCm.Remove(VehicleName);

	return true;
}

APawn* UAeroRuntimeOrchestrationSubsystem::ResolveSpawnedPawnByVehicleName(const FString& VehicleName) const
{
	if (GetWorld() == nullptr || VehicleName.TrimStartAndEnd().IsEmpty())
	{
		return nullptr;
	}

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (IsValid(SimModeActor))
	{
		if (APawn* ResolvedPawn = SimModeActor->resolveVehiclePawn(TCHAR_TO_UTF8(*VehicleName)); IsValid(ResolvedPawn))
		{
			return ResolvedPawn;
		}
	}

	for (TActorIterator<APawn> It(GetWorld()); It; ++It)
	{
		APawn* Pawn = *It;
		if (IsValid(Pawn) && Pawn->GetName().Equals(VehicleName, ESearchCase::CaseSensitive))
		{
			return Pawn;
		}
	}

	return nullptr;
}

bool UAeroRuntimeOrchestrationSubsystem::GetVehiclePose(
	const FString& VehicleName,
	FVector& OutWorldLocationCm,
	FRotator& OutWorldRotation,
	FString& OutError) const
{
	OutError.Reset();
	OutWorldLocationCm = FVector::ZeroVector;
	OutWorldRotation = FRotator::ZeroRotator;

	APawn* RuntimePawn = ResolveSpawnedPawnByVehicleName(VehicleName);
	if (!IsValid(RuntimePawn))
	{
		OutError = FString::Printf(TEXT("AirSim runtime pawn is unavailable for '%s'."), *VehicleName);
		return false;
	}

	OutWorldLocationCm = RuntimePawn->GetActorLocation();
	OutWorldRotation = RuntimePawn->GetActorRotation();
	return true;
}

UPedestrianWorldSubsystem* UAeroRuntimeOrchestrationSubsystem::ResolvePedestrianSubsystem(FString& OutError) const
{
	OutError.Reset();

	UPedestrianWorldSubsystem* PedSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UPedestrianWorldSubsystem>() : nullptr;
	if (PedSubsystem == nullptr)
	{
		OutError = TEXT("PedestrianWorldSubsystem unavailable.");
	}
	return PedSubsystem;
}

ASimModeBase* UAeroRuntimeOrchestrationSubsystem::ResolveSimModeActor() const
{
	if (ASimModeBase* SimModeActor = ASimModeBase::getSimMode())
	{
		if (SimModeActor->GetWorld() == GetWorld())
		{
			return SimModeActor;
		}
	}

	for (TActorIterator<ASimModeBase> It(GetWorld()); It; ++It)
	{
		return *It;
	}

	return nullptr;
}

bool UAeroRuntimeOrchestrationSubsystem::CancelTrackedMove(const FString& VehicleName, const bool bWaitForCompletion, FString& OutError)
{
	OutError.Reset();

	TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe> Job;
	{
		FScopeLock Lock(&MultirotorMoveJobsMutex);
		if (const TSharedPtr<FMultirotorMoveJobState, ESPMode::ThreadSafe>* ExistingJob = MultirotorMoveJobs.Find(VehicleName))
		{
			Job = *ExistingJob;
		}
	}

	if (!Job.IsValid())
	{
		return true;
	}

	bool bShouldCancel = false;
	{
		FScopeLock Lock(&Job->Mutex);
		bShouldCancel = Job->State == EAeroRuntimeMoveState::Running;
		Job->bCancelRequested = true;
		if (bShouldCancel)
		{
			Job->Message = TEXT("Move cancelled.");
		}
	}

	if (!bShouldCancel)
	{
		return true;
	}

	ASimModeBase* SimModeActor = ResolveSimModeActor();
	if (!IsValid(SimModeActor))
	{
		OutError = TEXT("AirSim SimMode is unavailable while cancelling the active multirotor move.");
		return false;
	}

	if (msr::airlib::VehicleApiBase* VehicleApi = SimModeActor->getApiProvider()->getVehicleApi(TCHAR_TO_UTF8(*VehicleName)))
	{
		VehicleApi->cancelLastTask();
	}

	if (bWaitForCompletion)
	{
		while (true)
		{
			bool bStillRunning = false;
			{
				FScopeLock Lock(&Job->Mutex);
				bStillRunning = Job->State == EAeroRuntimeMoveState::Running;
			}

			if (!bStillRunning)
			{
				break;
			}

			FPlatformProcess::SleepNoStats(0.01f);
		}
	}

	return true;
}
