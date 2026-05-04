#include "AeroFixedWorldCaptureCamera.h"

#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/World.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "ImageUtils.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RenderingThread.h"

namespace
{
const FSoftClassPath FixedWorldCaptureWeatherActorClassPath(TEXT("AActor'/AirSim/Weather/WeatherFX/WeatherActor.WeatherActor_C'"));
}

AAeroFixedWorldCaptureCamera::AAeroFixedWorldCaptureCamera()
{
	PrimaryActorTick.bCanEverTick = false;

	SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SetRootComponent(SceneRoot);

	PreviewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("PreviewCamera"));
	PreviewCamera->SetupAttachment(SceneRoot);
	PreviewCamera->SetFieldOfView(70.0f);

	SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
	SceneCapture->SetupAttachment(PreviewCamera);
	SceneCapture->bCaptureEveryFrame = false;
	SceneCapture->bCaptureOnMovement = false;
	SceneCapture->bAlwaysPersistRenderingState = true;
	SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	SceneCapture->FOVAngle = 70.0f;
	SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
	SceneCapture->ShowFlags.SetDepthOfField(false);
	SceneCapture->ShowFlags.SetMotionBlur(false);
}

void AAeroFixedWorldCaptureCamera::BeginPlay()
{
	Super::BeginPlay();
	EnsureWeatherFollower();
}

void AAeroFixedWorldCaptureCamera::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (IsValid(WeatherFollowerActor))
	{
		WeatherFollowerActor->Destroy();
		WeatherFollowerActor = nullptr;
	}
	Super::EndPlay(EndPlayReason);
}

bool AAeroFixedWorldCaptureCamera::EnsureRenderTarget(int32 Width, int32 Height, FString& OutError)
{
	if (Width <= 0 || Height <= 0)
	{
		OutError = TEXT("capture dimensions must be positive.");
		return false;
	}

	if (!IsValid(RenderTarget))
	{
		RenderTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("FixedWorldCaptureRenderTarget"));
		if (!IsValid(RenderTarget))
		{
			OutError = TEXT("failed to allocate render target.");
			return false;
		}
		RenderTarget->ClearColor = FLinearColor::Black;
		RenderTarget->TargetGamma = 2.2f;
		RenderTarget->bAutoGenerateMips = false;
	}

	if (RenderTarget->SizeX != Width || RenderTarget->SizeY != Height)
	{
		RenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, false);
		RenderTarget->UpdateResourceImmediate(true);
	}

	SceneCapture->TextureTarget = RenderTarget;
	return true;
}

void AAeroFixedWorldCaptureCamera::EnsureWeatherFollower()
{
	if (IsValid(WeatherFollowerActor))
	{
		return;
	}

	UWorld* World = GetWorld();
	if (World == nullptr || !World->IsGameWorld())
	{
		return;
	}

	UClass* WeatherActorClass = FixedWorldCaptureWeatherActorClassPath.TryLoadClass<AActor>();
	if (WeatherActorClass == nullptr)
	{
		UE_LOG(LogTemp, Warning, TEXT("FixedWorldCaptureCamera could not load weather actor class."));
		return;
	}

	FActorSpawnParameters SpawnInfo;
	SpawnInfo.Owner = this;
	SpawnInfo.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	AActor* SpawnedWeatherActor = World->SpawnActor<AActor>(WeatherActorClass, GetActorLocation(), GetActorRotation(), SpawnInfo);
	if (!IsValid(SpawnedWeatherActor))
	{
		UE_LOG(LogTemp, Warning, TEXT("FixedWorldCaptureCamera failed to spawn weather follower actor."));
		return;
	}

	SpawnedWeatherActor->AttachToActor(this, FAttachmentTransformRules(EAttachmentRule::SnapToTarget, true));
	SpawnedWeatherActor->SetActorScale3D(FVector(FMath::Max(0.1f, WeatherFollowerScale)));
	WeatherFollowerActor = SpawnedWeatherActor;
}

bool AAeroFixedWorldCaptureCamera::CaptureRgbToDisk(
	const FString& AbsoluteOutputPath,
	int32 Width,
	int32 Height,
	float FovDegrees,
	FString& OutError,
	int32& OutCapturedWidth,
	int32& OutCapturedHeight)
{
	OutCapturedWidth = 0;
	OutCapturedHeight = 0;

	if (!IsValid(SceneCapture) || !IsValid(PreviewCamera))
	{
		OutError = TEXT("camera components are unavailable.");
		return false;
	}

	if (!EnsureRenderTarget(Width, Height, OutError))
	{
		return false;
	}

	if (FovDegrees > 1.0f)
	{
		PreviewCamera->SetFieldOfView(FovDegrees);
		SceneCapture->FOVAngle = FovDegrees;
	}

	const FString Directory = FPaths::GetPath(AbsoluteOutputPath);
	if (!Directory.IsEmpty() && !IFileManager::Get().MakeDirectory(*Directory, true))
	{
		OutError = FString::Printf(TEXT("failed to create capture directory: %s"), *Directory);
		return false;
	}

	SceneCapture->CaptureScene();
	FlushRenderingCommands();

	FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
	if (RenderTargetResource == nullptr)
	{
		OutError = TEXT("render target resource is unavailable.");
		return false;
	}

	TArray<FColor> Bitmap;
	FReadSurfaceDataFlags ReadFlags(RCM_UNorm);
	ReadFlags.SetLinearToGamma(false);
	if (!RenderTargetResource->ReadPixels(Bitmap, ReadFlags))
	{
		OutError = TEXT("ReadPixels failed.");
		return false;
	}

	if (Bitmap.Num() != Width * Height)
	{
		OutError = FString::Printf(TEXT("unexpected pixel count: expected %d got %d."), Width * Height, Bitmap.Num());
		return false;
	}

	TArray<uint8> PngBytes;
	FImageUtils::CompressImageArray(Width, Height, Bitmap, PngBytes);
	if (PngBytes.Num() <= 0)
	{
		OutError = TEXT("PNG compression failed.");
		return false;
	}

	if (!FFileHelper::SaveArrayToFile(PngBytes, *AbsoluteOutputPath))
	{
		OutError = FString::Printf(TEXT("failed to save PNG: %s"), *AbsoluteOutputPath);
		return false;
	}

	OutCapturedWidth = Width;
	OutCapturedHeight = Height;
	return true;
}
