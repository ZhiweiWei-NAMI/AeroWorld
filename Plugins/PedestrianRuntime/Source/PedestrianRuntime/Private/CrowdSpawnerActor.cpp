#include "CrowdSpawnerActor.h"

#include "Components/BoxComponent.h"
#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "PedestrianRuntimeLog.h"
#include "PedestrianWorldSubsystem.h"

ACrowdSpawnerActor::ACrowdSpawnerActor()
{
	PrimaryActorTick.bCanEverTick = false;

	SpawnZone = CreateDefaultSubobject<UBoxComponent>(TEXT("SpawnZone"));
	SpawnZone->SetMobility(EComponentMobility::Static);
	SpawnZone->SetBoxExtent(FVector(500.0f, 500.0f, 200.0f));
	SpawnZone->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	RootComponent = SpawnZone;
}

void ACrowdSpawnerActor::BeginPlay()
{
	Super::BeginPlay();

	if (bAutoSpawnOnBeginPlay)
	{
		SpawnCrowdNow();
	}
}

FCrowdSpawnResult ACrowdSpawnerActor::SpawnCrowdNow()
{
	FCrowdSpawnResult Result;
	Result.GroupId = GroupId;
	Result.Seed = Seed;

	UWorld* World = GetWorld();
	if (World == nullptr || !World->IsGameWorld())
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CrowdSpawner '%s' spawn skipped: world is unavailable or not game world."), *GetName());
		return Result;
	}

	UPedestrianWorldSubsystem* Subsystem = World->GetSubsystem<UPedestrianWorldSubsystem>();
	if (Subsystem == nullptr)
	{
		UE_LOG(LogPedestrianRuntime, Warning, TEXT("CrowdSpawner '%s' spawn failed: PedestrianWorldSubsystem unavailable."), *GetName());
		return Result;
	}

	return Subsystem->SpawnCrowd(BuildRequest());
}

bool ACrowdSpawnerActor::ClearSpawnedCrowd()
{
	UWorld* World = GetWorld();
	if (World == nullptr || !World->IsGameWorld())
	{
		return false;
	}

	UPedestrianWorldSubsystem* Subsystem = World->GetSubsystem<UPedestrianWorldSubsystem>();
	return Subsystem != nullptr && Subsystem->ClearCrowdGroup(GroupId);
}

FCrowdSpawnRequest ACrowdSpawnerActor::BuildRequest() const
{
	FCrowdSpawnRequest Request;
	Request.GroupId = GroupId;
	Request.Count = FMath::Max(0, Count);
	Request.Seed = Seed;
	Request.AppearancePool = AppearancePool;
	Request.RoleProfile = RoleProfile;
	Request.YawPolicy = YawPolicy;
	Request.FixedYawDeg = FixedYawDeg;
	Request.CollisionHandling = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
	Request.SpawnOrigin = GetActorLocation();
	Request.SpawnBoxExtent = IsValid(SpawnZone) ? SpawnZone->GetScaledBoxExtent() : FVector::ZeroVector;
	return Request;
}
