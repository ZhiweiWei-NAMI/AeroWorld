#include "AeroTriggerZoneBase.h"

#include "AeroSemanticBindingComponent.h"
#include "AeroTriggerRelayComponent.h"
#include "Components/BoxComponent.h"
#include "Components/SceneComponent.h"
#include "Components/SphereComponent.h"

namespace
{
void ConfigureTriggerPrimitive(UPrimitiveComponent* Primitive, bool bEnable)
{
	if (!IsValid(Primitive))
	{
		return;
	}

	Primitive->SetCollisionEnabled(bEnable ? ECollisionEnabled::QueryOnly : ECollisionEnabled::NoCollision);
	Primitive->SetCollisionResponseToAllChannels(ECR_Ignore);
	Primitive->SetCollisionResponseToChannel(ECC_Pawn, ECR_Overlap);
	Primitive->SetCollisionResponseToChannel(ECC_Vehicle, ECR_Overlap);
	Primitive->SetCollisionObjectType(ECC_WorldDynamic);
	Primitive->SetGenerateOverlapEvents(bEnable);
	Primitive->SetHiddenInGame(true);
	Primitive->SetVisibility(false, true);
}
}

AAeroTriggerZoneBase::AAeroTriggerZoneBase()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.SetTickFunctionEnable(false);

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	RootComponent = SceneRoot;

	BoxTrigger = CreateDefaultSubobject<UBoxComponent>(TEXT("BoxTrigger"));
	BoxTrigger->SetupAttachment(SceneRoot);

	SphereTrigger = CreateDefaultSubobject<USphereComponent>(TEXT("SphereTrigger"));
	SphereTrigger->SetupAttachment(SceneRoot);

	SemanticBinding = CreateDefaultSubobject<UAeroSemanticBindingComponent>(TEXT("SemanticBinding"));
	TriggerRelay = CreateDefaultSubobject<UAeroTriggerRelayComponent>(TEXT("TriggerRelay"));

	SetActorHiddenInGame(true);
}

void AAeroTriggerZoneBase::BeginPlay()
{
	Super::BeginPlay();
	RefreshShapeComponents();
}

void AAeroTriggerZoneBase::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	if (ShapeConfig.ShapeKind != EAeroTriggerShapeKind::PolygonPrism || !IsValid(BoxTrigger) || !IsValid(TriggerRelay))
	{
		return;
	}

	TArray<AActor*> OverlappingActors;
	BoxTrigger->GetOverlappingActors(OverlappingActors);

	TSet<TObjectPtr<AActor>> CurrentInsideActors;
	for (AActor* Actor : OverlappingActors)
	{
		if (IsValid(Actor) && Actor != this && IsWorldLocationInsideZone(Actor->GetActorLocation()))
		{
			CurrentInsideActors.Add(Actor);
			if (!PolygonInsideActors.Contains(Actor))
			{
				TriggerRelay->EmitOverlapEnter(Actor);
			}
		}
	}

	for (AActor* ExistingActor : PolygonInsideActors)
	{
		if (!CurrentInsideActors.Contains(ExistingActor))
		{
			TriggerRelay->EmitOverlapExit(ExistingActor);
		}
	}

	PolygonInsideActors = MoveTemp(CurrentInsideActors);
}

void AAeroTriggerZoneBase::ConfigureShape(const FAeroTriggerShapeConfig& InShapeConfig)
{
	ShapeConfig = InShapeConfig;
	RefreshShapeComponents();
}

bool AAeroTriggerZoneBase::IsWorldLocationInsideZone(const FVector& WorldLocationCm) const
{
	const FVector LocalPoint = GetActorTransform().InverseTransformPosition(WorldLocationCm);
	switch (ShapeConfig.ShapeKind)
	{
	case EAeroTriggerShapeKind::Box:
		return FMath::Abs(LocalPoint.X) <= ShapeConfig.BoxExtentCm.X &&
			FMath::Abs(LocalPoint.Y) <= ShapeConfig.BoxExtentCm.Y &&
			FMath::Abs(LocalPoint.Z) <= ShapeConfig.BoxExtentCm.Z;
	case EAeroTriggerShapeKind::Sphere:
		return LocalPoint.SizeSquared() <= FMath::Square(ShapeConfig.SphereRadiusCm);
	case EAeroTriggerShapeKind::PolygonPrism:
		return IsLocalPointInsidePolygon(LocalPoint);
	default:
		return false;
	}
}

EAeroTriggerShapeKind AAeroTriggerZoneBase::GetShapeKind() const
{
	return ShapeConfig.ShapeKind;
}

UPrimitiveComponent* AAeroTriggerZoneBase::GetActiveTriggerComponent() const
{
	switch (ShapeConfig.ShapeKind)
	{
	case EAeroTriggerShapeKind::Sphere:
		return SphereTrigger;
	case EAeroTriggerShapeKind::PolygonPrism:
	case EAeroTriggerShapeKind::Box:
		return BoxTrigger;
	default:
		return nullptr;
	}
}

const FAeroTriggerShapeConfig& AAeroTriggerZoneBase::GetShapeConfig() const
{
	return ShapeConfig;
}

void AAeroTriggerZoneBase::RefreshShapeComponents()
{
	ConfigureTriggerPrimitive(BoxTrigger, false);
	ConfigureTriggerPrimitive(SphereTrigger, false);
	PolygonInsideActors.Reset();

	if (!IsValid(TriggerRelay))
	{
		return;
	}

	switch (ShapeConfig.ShapeKind)
	{
	case EAeroTriggerShapeKind::Box:
		if (IsValid(BoxTrigger))
		{
			BoxTrigger->SetRelativeLocation(FVector::ZeroVector);
			BoxTrigger->SetBoxExtent(ShapeConfig.BoxExtentCm);
			ConfigureTriggerPrimitive(BoxTrigger, true);
			TriggerRelay->SetTriggerComponent(BoxTrigger);
		}
		PrimaryActorTick.SetTickFunctionEnable(false);
		break;
	case EAeroTriggerShapeKind::Sphere:
		if (IsValid(SphereTrigger))
		{
			SphereTrigger->SetRelativeLocation(FVector::ZeroVector);
			SphereTrigger->SetSphereRadius(ShapeConfig.SphereRadiusCm);
			ConfigureTriggerPrimitive(SphereTrigger, true);
			TriggerRelay->SetTriggerComponent(SphereTrigger);
		}
		PrimaryActorTick.SetTickFunctionEnable(false);
		break;
	case EAeroTriggerShapeKind::PolygonPrism:
		if (IsValid(BoxTrigger))
		{
			FVector2D Min2D(FLT_MAX, FLT_MAX);
			FVector2D Max2D(-FLT_MAX, -FLT_MAX);
			for (const FVector2D& Vertex : ShapeConfig.PolygonVerticesCm)
			{
				Min2D.X = FMath::Min(Min2D.X, Vertex.X);
				Min2D.Y = FMath::Min(Min2D.Y, Vertex.Y);
				Max2D.X = FMath::Max(Max2D.X, Vertex.X);
				Max2D.Y = FMath::Max(Max2D.Y, Vertex.Y);
			}

			const FVector Extent(
				FMath::Max(50.0f, (Max2D.X - Min2D.X) * 0.5f),
				FMath::Max(50.0f, (Max2D.Y - Min2D.Y) * 0.5f),
				FMath::Max(50.0f, (ShapeConfig.PolygonMaxZCm - ShapeConfig.PolygonMinZCm) * 0.5f));
			BoxTrigger->SetRelativeLocation(FVector((Min2D.X + Max2D.X) * 0.5f, (Min2D.Y + Max2D.Y) * 0.5f, 0.0f));
			BoxTrigger->SetBoxExtent(Extent);
			ConfigureTriggerPrimitive(BoxTrigger, true);
			TriggerRelay->SetTriggerComponent(BoxTrigger);
		}
		PrimaryActorTick.SetTickFunctionEnable(true);
		break;
	default:
		TriggerRelay->SetTriggerComponent(nullptr);
		PrimaryActorTick.SetTickFunctionEnable(false);
		break;
	}
}

bool AAeroTriggerZoneBase::IsLocalPointInsidePolygon(const FVector& LocalPointCm) const
{
	if (LocalPointCm.Z < ShapeConfig.PolygonMinZCm || LocalPointCm.Z > ShapeConfig.PolygonMaxZCm || ShapeConfig.PolygonVerticesCm.Num() < 3)
	{
		return false;
	}

	bool bInside = false;
	const FVector2D TestPoint(LocalPointCm.X, LocalPointCm.Y);
	for (int32 Index = 0, PrevIndex = ShapeConfig.PolygonVerticesCm.Num() - 1; Index < ShapeConfig.PolygonVerticesCm.Num(); PrevIndex = Index++)
	{
		const FVector2D& VertexA = ShapeConfig.PolygonVerticesCm[Index];
		const FVector2D& VertexB = ShapeConfig.PolygonVerticesCm[PrevIndex];
		const bool bCrosses = ((VertexA.Y > TestPoint.Y) != (VertexB.Y > TestPoint.Y)) &&
			(TestPoint.X < (VertexB.X - VertexA.X) * (TestPoint.Y - VertexA.Y) / FMath::Max(KINDA_SMALL_NUMBER, VertexB.Y - VertexA.Y) + VertexA.X);
		if (bCrosses)
		{
			bInside = !bInside;
		}
	}

	return bInside;
}
