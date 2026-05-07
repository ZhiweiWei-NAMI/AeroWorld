#include "AeroFixedWorldCaptureCamera.h"

#include "AeroSemanticBindingComponent.h"
#include "Annotation/AnnotationComponent.h"
#include "Camera/CameraComponent.h"
#include "Components/MeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/World.h"
#include "Engine/TextureRenderTarget2D.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "ImageUtils.h"
#include "Misc/Char.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RenderingThread.h"

#include <initializer_list>
#include <limits>

namespace
{
const FSoftClassPath FixedWorldCaptureWeatherActorClassPath(TEXT("AActor'/AirSim/Weather/WeatherFX/WeatherActor.WeatherActor_C'"));

struct FAeroSemanticPaletteEntry
{
	const TCHAR* ClassName;
	FColor Color;
};

const TArray<FAeroSemanticPaletteEntry>& AeroSemanticPalette()
{
	static const TArray<FAeroSemanticPaletteEntry> Palette = {
		{TEXT("background"), FColor(0, 0, 0)},
		{TEXT("static_map"), FColor(90, 90, 90)},
		{TEXT("uav"), FColor(255, 0, 0)},
		{TEXT("vehicle"), FColor(0, 128, 255)},
		{TEXT("pedestrian"), FColor(255, 255, 0)},
		{TEXT("roadwork_prop"), FColor(255, 128, 0)},
		{TEXT("traffic_control"), FColor(0, 255, 255)},
		{TEXT("facility"), FColor(128, 0, 255)},
		{TEXT("hazard_trigger"), FColor(255, 0, 255)},
	};
	return Palette;
}

FColor SemanticColorForClass(const FString& ClassName)
{
	for (const FAeroSemanticPaletteEntry& Entry : AeroSemanticPalette())
	{
		if (ClassName.Equals(Entry.ClassName, ESearchCase::IgnoreCase))
		{
			return Entry.Color;
		}
	}
	return FColor(90, 90, 90);
}

bool ContainsAnyToken(const FString& Haystack, std::initializer_list<const TCHAR*> Tokens)
{
	for (const TCHAR* Token : Tokens)
	{
		if (Haystack.Contains(Token))
		{
			return true;
		}
	}
	return false;
}

FString SemanticTextForActor(const AActor* Actor)
{
	FString Text = Actor != nullptr ? Actor->GetName() : FString();
#if WITH_EDITOR
	if (Actor != nullptr)
	{
		Text += TEXT(" ");
		Text += Actor->GetActorLabel();
	}
#endif
	if (Actor != nullptr)
	{
		for (const FName& Tag : Actor->Tags)
		{
			Text += TEXT(" ");
			Text += Tag.ToString();
		}

		if (const UAeroSemanticBindingComponent* Binding = Actor->FindComponentByClass<UAeroSemanticBindingComponent>())
		{
			Text += TEXT(" ");
			Text += Binding->EntityId;
			Text += TEXT(" ");
			Text += Binding->InstanceId;
			Text += TEXT(" ");
			Text += Binding->LogicalAssetId;
			Text += TEXT(" ");
			Text += Binding->LabelClass;
			Text += TEXT(" ");
			Text += Binding->WorldLayerType;
			Text += TEXT(" ");
			Text += Binding->ZoneKind;
			for (const FString& Tag : Binding->Tags)
			{
				Text += TEXT(" ");
				Text += Tag;
			}
		}
	}
	return Text.ToLower();
}

FString SemanticClassForActor(const AActor* Actor)
{
	const FString Text = SemanticTextForActor(Actor);
	if (ContainsAnyToken(Text, {TEXT("trigger."), TEXT("trigger_"), TEXT("hazard"), TEXT("no_fly"), TEXT("nfz"), TEXT("geofence")}))
	{
		return TEXT("hazard_trigger");
	}
	if (ContainsAnyToken(Text, {TEXT("pedestrian"), TEXT("walker"), TEXT("crowd"), TEXT("ped_")}))
	{
		return TEXT("pedestrian");
	}
	if (ContainsAnyToken(Text, {TEXT("uav"), TEXT("drone"), TEXT("multirotor"), TEXT("flyingpawn"), TEXT("cv_pawn")}))
	{
		return TEXT("uav");
	}
	if (ContainsAnyToken(Text, {TEXT("vehicle."), TEXT("vehicle_"), TEXT("ambulance"), TEXT("police_suv"), TEXT("suv"), TEXT("boxcar"), TEXT("car_")}))
	{
		return TEXT("vehicle");
	}
	if (ContainsAnyToken(Text, {TEXT("roadwork"), TEXT("barrier"), TEXT("cone"), TEXT("construction_fence")}))
	{
		return TEXT("roadwork_prop");
	}
	if (ContainsAnyToken(Text, {TEXT("traffic_control"), TEXT("signal_light"), TEXT("police_sign"), TEXT("police_tape"), TEXT("traffic_signal")}))
	{
		return TEXT("traffic_control");
	}
	if (ContainsAnyToken(Text, {TEXT("facility."), TEXT("landing_pad"), TEXT("base_tower"), TEXT("charger"), TEXT("fixed_world_capture")}))
	{
		return TEXT("facility");
	}
	return TEXT("static_map");
}

void ConfigureSemanticCaptureShowFlags(FEngineShowFlags& ShowFlags)
{
	ShowFlags.SetMaterials(false);
	ShowFlags.SetLighting(false);
	ShowFlags.SetBSPTriangles(true);
	ShowFlags.SetPostProcessing(false);
	ShowFlags.SetHMDDistortion(false);
	ShowFlags.SetTonemapper(false);
	ShowFlags.SetEyeAdaptation(false);
	ShowFlags.SetFog(false);
	ShowFlags.SetPaper2DSprites(false);
	ShowFlags.SetBloom(false);
	ShowFlags.SetMotionBlur(false);
	ShowFlags.SetSkyLighting(false);
	ShowFlags.SetVisualizeSkyAtmosphere(false);
	ShowFlags.SetAmbientOcclusion(false);
	ShowFlags.SetAtmosphere(false);
	ShowFlags.SetInstancedFoliage(false);
	ShowFlags.SetInstancedGrass(false);
	ShowFlags.SetTextRender(false);
	ShowFlags.SetTemporalAA(false);
	ShowFlags.SetDecals(false);
}

struct FSceneCaptureStateGuard
{
	explicit FSceneCaptureStateGuard(USceneCaptureComponent2D* InCapture)
		: Capture(InCapture)
		, CaptureSource(InCapture != nullptr ? InCapture->CaptureSource : ESceneCaptureSource::SCS_FinalColorLDR)
		, PrimitiveRenderMode(InCapture != nullptr ? InCapture->PrimitiveRenderMode : ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives)
		, ShowFlags(InCapture != nullptr ? InCapture->ShowFlags : FEngineShowFlags(ESFIM_Game))
		, ShowOnlyComponents(InCapture != nullptr ? InCapture->ShowOnlyComponents : TArray<TWeakObjectPtr<UPrimitiveComponent>>())
		, HiddenComponents(InCapture != nullptr ? InCapture->HiddenComponents : TArray<TWeakObjectPtr<UPrimitiveComponent>>())
	{
	}

	~FSceneCaptureStateGuard()
	{
		if (Capture.IsValid())
		{
			USceneCaptureComponent2D* CapturePtr = Capture.Get();
			CapturePtr->CaptureSource = CaptureSource;
			CapturePtr->PrimitiveRenderMode = PrimitiveRenderMode;
			CapturePtr->ShowFlags = ShowFlags;
			CapturePtr->ShowOnlyComponents = ShowOnlyComponents;
			CapturePtr->HiddenComponents = HiddenComponents;
		}
	}

	TWeakObjectPtr<USceneCaptureComponent2D> Capture;
	ESceneCaptureSource CaptureSource;
	ESceneCapturePrimitiveRenderMode PrimitiveRenderMode;
	FEngineShowFlags ShowFlags;
	TArray<TWeakObjectPtr<UPrimitiveComponent>> ShowOnlyComponents;
	TArray<TWeakObjectPtr<UPrimitiveComponent>> HiddenComponents;
};

bool EnsureOutputDirectory(const FString& AbsoluteOutputPath, FString& OutError)
{
	const FString Directory = FPaths::GetPath(AbsoluteOutputPath);
	if (!Directory.IsEmpty() && !IFileManager::Get().MakeDirectory(*Directory, true))
	{
		OutError = FString::Printf(TEXT("failed to create capture directory: %s"), *Directory);
		return false;
	}
	return true;
}

bool ReadRenderTargetColorPixels(UTextureRenderTarget2D* RenderTarget, int32 Width, int32 Height, TArray<FColor>& OutBitmap, FString& OutError)
{
	if (!IsValid(RenderTarget))
	{
		OutError = TEXT("render target is unavailable.");
		return false;
	}

	FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
	if (RenderTargetResource == nullptr)
	{
		OutError = TEXT("render target resource is unavailable.");
		return false;
	}

	FReadSurfaceDataFlags ReadFlags(RCM_UNorm);
	ReadFlags.SetLinearToGamma(false);
	if (!RenderTargetResource->ReadPixels(OutBitmap, ReadFlags))
	{
		OutError = TEXT("ReadPixels failed.");
		return false;
	}

	if (OutBitmap.Num() != Width * Height)
	{
		OutError = FString::Printf(TEXT("unexpected pixel count: expected %d got %d."), Width * Height, OutBitmap.Num());
		return false;
	}
	return true;
}

void DestroySemanticAnnotations(TArray<TWeakObjectPtr<UAnnotationComponent>>& AnnotationComponents)
{
	for (TWeakObjectPtr<UAnnotationComponent>& ComponentPtr : AnnotationComponents)
	{
		if (UAnnotationComponent* Component = ComponentPtr.Get())
		{
			Component->DestroyComponent();
		}
	}
	AnnotationComponents.Reset();
}

void CreateSemanticAnnotations(UWorld* World, const AActor* CaptureActor, const AActor* WeatherFollowerActor, TArray<TWeakObjectPtr<UAnnotationComponent>>& OutAnnotationComponents)
{
	if (World == nullptr)
	{
		return;
	}

	int32 AnnotationIndex = 0;
	for (TActorIterator<AActor> ActorIt(World); ActorIt; ++ActorIt)
	{
		AActor* Actor = *ActorIt;
		if (!IsValid(Actor) || Actor == CaptureActor || Actor == WeatherFollowerActor || Actor->IsA<AAeroFixedWorldCaptureCamera>() || Actor->IsHidden())
		{
			continue;
		}

		const FColor SemanticColor = SemanticColorForClass(SemanticClassForActor(Actor));
		TArray<UMeshComponent*> MeshComponents;
		Actor->GetComponents<UMeshComponent>(MeshComponents);
		for (UMeshComponent* MeshComponent : MeshComponents)
		{
			if (!IsValid(MeshComponent) || !MeshComponent->IsRegistered() || !MeshComponent->IsVisible())
			{
				continue;
			}

			const FName AnnotationName(*FString::Printf(TEXT("AeroSemanticCapture_%d"), AnnotationIndex++));
			UAnnotationComponent* AnnotationComponent = NewObject<UAnnotationComponent>(MeshComponent, AnnotationName);
			if (!IsValid(AnnotationComponent))
			{
				continue;
			}

			AnnotationComponent->SetupAttachment(MeshComponent);
			AnnotationComponent->RegisterComponent();
			AnnotationComponent->SetAnnotationColor(SemanticColor);
			AnnotationComponent->SetVisibleInSceneCaptureOnly(true);
			AnnotationComponent->SetVisibleInRayTracing(false);
			AnnotationComponent->bVisibleInReflectionCaptures = false;
			AnnotationComponent->bAffectDynamicIndirectLighting = false;
			AnnotationComponent->bAffectDistanceFieldLighting = false;
			AnnotationComponent->bVisibleInRealTimeSkyCaptures = false;
			AnnotationComponent->bRenderInMainPass = false;
			AnnotationComponent->MarkRenderStateDirty();
			OutAnnotationComponents.Add(AnnotationComponent);
		}
	}
}
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

bool AAeroFixedWorldCaptureCamera::EnsureRenderTarget(int32 Width, int32 Height, bool bFloatRenderTarget, FString& OutError)
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
		RenderTarget->bAutoGenerateMips = false;
	}

	if (RenderTargetWidth != Width || RenderTargetHeight != Height || bRenderTargetFloat != bFloatRenderTarget)
	{
		RenderTarget->TargetGamma = bFloatRenderTarget ? 1.0f : 2.2f;
		RenderTarget->ClearColor = FLinearColor::Black;
		RenderTarget->InitCustomFormat(Width, Height, bFloatRenderTarget ? PF_FloatRGBA : PF_B8G8R8A8, bFloatRenderTarget);
		RenderTarget->UpdateResourceImmediate(true);
		RenderTargetWidth = Width;
		RenderTargetHeight = Height;
		bRenderTargetFloat = bFloatRenderTarget;
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
	FAeroFixedWorldCaptureStats Stats;
	const bool bSuccess = CaptureToDisk(TEXT("rgb"), AbsoluteOutputPath, Width, Height, FovDegrees, OutError, Stats);
	OutCapturedWidth = Stats.CapturedWidth;
	OutCapturedHeight = Stats.CapturedHeight;
	return bSuccess;
}

bool AAeroFixedWorldCaptureCamera::CaptureToDisk(
	const FString& Modality,
	const FString& AbsoluteOutputPath,
	int32 Width,
	int32 Height,
	float FovDegrees,
	FString& OutError,
	FAeroFixedWorldCaptureStats& OutStats)
{
	OutStats = FAeroFixedWorldCaptureStats();
	if (!IsValid(SceneCapture) || !IsValid(PreviewCamera))
	{
		OutError = TEXT("camera components are unavailable.");
		return false;
	}

	const FString NormalizedModality = Modality.TrimStartAndEnd().ToLower();
	const bool bDepthCapture = NormalizedModality.Equals(TEXT("depth"), ESearchCase::IgnoreCase);
	if (!NormalizedModality.Equals(TEXT("rgb"), ESearchCase::IgnoreCase)
		&& !NormalizedModality.Equals(TEXT("depth"), ESearchCase::IgnoreCase)
		&& !NormalizedModality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		OutError = FString::Printf(TEXT("unsupported fixed world capture modality '%s'."), *Modality);
		return false;
	}

	if (!EnsureRenderTarget(Width, Height, bDepthCapture, OutError))
	{
		return false;
	}

	if (FovDegrees > 1.0f)
	{
		PreviewCamera->SetFieldOfView(FovDegrees);
		SceneCapture->FOVAngle = FovDegrees;
	}

	if (!EnsureOutputDirectory(AbsoluteOutputPath, OutError))
	{
		return false;
	}

	FSceneCaptureStateGuard CaptureStateGuard(SceneCapture);
	if (NormalizedModality.Equals(TEXT("depth"), ESearchCase::IgnoreCase))
	{
		SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
		SceneCapture->ShowFlags.SetPostProcessing(false);
		SceneCapture->ShowFlags.SetMotionBlur(false);
		SceneCapture->ShowFlags.SetDepthOfField(false);
		return CaptureDepthNpyToDisk(AbsoluteOutputPath, Width, Height, OutError, OutStats);
	}
	if (NormalizedModality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
		ConfigureSemanticCaptureShowFlags(SceneCapture->ShowFlags);
		return CaptureSemanticPngToDisk(AbsoluteOutputPath, Width, Height, OutError, OutStats);
	}

	SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	SceneCapture->ShowFlags.SetDepthOfField(false);
	SceneCapture->ShowFlags.SetMotionBlur(false);
	return CaptureColorPngToDisk(AbsoluteOutputPath, Width, Height, OutError, OutStats);
}

bool AAeroFixedWorldCaptureCamera::CaptureColorPngToDisk(
	const FString& AbsoluteOutputPath,
	int32 Width,
	int32 Height,
	FString& OutError,
	FAeroFixedWorldCaptureStats& OutStats)
{
	SceneCapture->CaptureScene();
	FlushRenderingCommands();

	TArray<FColor> Bitmap;
	if (!ReadRenderTargetColorPixels(RenderTarget, Width, Height, Bitmap, OutError))
	{
		return false;
	}

	TArray64<uint8> PngBytes;
	FImageUtils::PNGCompressImageArray(Width, Height, TArrayView64<const FColor>(Bitmap.GetData(), Bitmap.Num()), PngBytes);
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

	OutStats.CapturedWidth = Width;
	OutStats.CapturedHeight = Height;
	OutStats.OutputFormat = TEXT("png");
	return true;
}

bool AAeroFixedWorldCaptureCamera::CaptureDepthNpyToDisk(
	const FString& AbsoluteOutputPath,
	int32 Width,
	int32 Height,
	FString& OutError,
	FAeroFixedWorldCaptureStats& OutStats)
{
	SceneCapture->CaptureScene();
	FlushRenderingCommands();

	FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
	if (RenderTargetResource == nullptr)
	{
		OutError = TEXT("render target resource is unavailable.");
		return false;
	}

	TArray<FFloat16Color> FloatPixels;
	if (!RenderTargetResource->ReadFloat16Pixels(FloatPixels))
	{
		OutError = TEXT("ReadFloat16Pixels failed.");
		return false;
	}

	if (FloatPixels.Num() != Width * Height)
	{
		OutError = FString::Printf(TEXT("unexpected depth pixel count: expected %d got %d."), Width * Height, FloatPixels.Num());
		return false;
	}

	TArray<float> DepthMeters;
	DepthMeters.SetNumUninitialized(FloatPixels.Num());
	float DepthMinM = TNumericLimits<float>::Max();
	float DepthMaxM = -TNumericLimits<float>::Max();
	int32 ValidCount = 0;
	int32 InvalidCount = 0;
	for (int32 Index = 0; Index < FloatPixels.Num(); ++Index)
	{
		const float DepthCm = FloatPixels[Index].R.GetFloat();
		const float DepthM = DepthCm / 100.0f;
		if (FMath::IsFinite(DepthM) && DepthM > 0.0f && DepthM < 100000.0f)
		{
			DepthMeters[Index] = DepthM;
			DepthMinM = FMath::Min(DepthMinM, DepthM);
			DepthMaxM = FMath::Max(DepthMaxM, DepthM);
			++ValidCount;
		}
		else
		{
			DepthMeters[Index] = std::numeric_limits<float>::quiet_NaN();
			++InvalidCount;
		}
	}
	if (ValidCount == 0)
	{
		DepthMinM = 0.0f;
		DepthMaxM = 0.0f;
	}

	FString Header = FString::Printf(TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }"), Height, Width);
	FTCHARToUTF8 HeaderUtf8(*Header);
	TArray<uint8> HeaderBytes;
	HeaderBytes.Append(reinterpret_cast<const uint8*>(HeaderUtf8.Get()), HeaderUtf8.Length());
	while (((10 + HeaderBytes.Num() + 1) % 16) != 0)
	{
		HeaderBytes.Add(static_cast<uint8>(' '));
	}
	HeaderBytes.Add(static_cast<uint8>('\n'));
	if (HeaderBytes.Num() > MAX_uint16)
	{
		OutError = TEXT("npy header is too large.");
		return false;
	}

	TArray<uint8> FileBytes;
	FileBytes.Reserve(10 + HeaderBytes.Num() + DepthMeters.Num() * sizeof(float));
	const uint8 Magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
	FileBytes.Append(Magic, UE_ARRAY_COUNT(Magic));
	const uint16 HeaderLength = static_cast<uint16>(HeaderBytes.Num());
	FileBytes.Add(static_cast<uint8>(HeaderLength & 0xff));
	FileBytes.Add(static_cast<uint8>((HeaderLength >> 8) & 0xff));
	FileBytes.Append(HeaderBytes);
	FileBytes.Append(reinterpret_cast<const uint8*>(DepthMeters.GetData()), DepthMeters.Num() * sizeof(float));

	if (!FFileHelper::SaveArrayToFile(FileBytes, *AbsoluteOutputPath))
	{
		OutError = FString::Printf(TEXT("failed to save depth NPY: %s"), *AbsoluteOutputPath);
		return false;
	}

	OutStats.CapturedWidth = Width;
	OutStats.CapturedHeight = Height;
	OutStats.OutputFormat = TEXT("npy_float32_m");
	OutStats.bDepthUnitMeters = true;
	OutStats.DepthMinM = DepthMinM;
	OutStats.DepthMaxM = DepthMaxM;
	OutStats.DepthValidCount = ValidCount;
	OutStats.DepthInvalidCount = InvalidCount;
	return true;
}

bool AAeroFixedWorldCaptureCamera::CaptureSemanticPngToDisk(
	const FString& AbsoluteOutputPath,
	int32 Width,
	int32 Height,
	FString& OutError,
	FAeroFixedWorldCaptureStats& OutStats)
{
	TArray<TWeakObjectPtr<UAnnotationComponent>> AnnotationComponents;
	CreateSemanticAnnotations(GetWorld(), this, WeatherFollowerActor.Get(), AnnotationComponents);
	SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
	SceneCapture->ShowOnlyComponents.Empty();
	for (const TWeakObjectPtr<UAnnotationComponent>& AnnotationPtr : AnnotationComponents)
	{
		if (UAnnotationComponent* AnnotationComponent = AnnotationPtr.Get())
		{
			SceneCapture->ShowOnlyComponents.Add(AnnotationComponent);
		}
	}
	FlushRenderingCommands();

	SceneCapture->CaptureScene();
	FlushRenderingCommands();

	TArray<FColor> Bitmap;
	const bool bReadOk = ReadRenderTargetColorPixels(RenderTarget, Width, Height, Bitmap, OutError);

	DestroySemanticAnnotations(AnnotationComponents);
	FlushRenderingCommands();

	if (!bReadOk)
	{
		return false;
	}

	TArray64<uint8> PngBytes;
	FImageUtils::PNGCompressImageArray(Width, Height, TArrayView64<const FColor>(Bitmap.GetData(), Bitmap.Num()), PngBytes);
	if (PngBytes.Num() <= 0)
	{
		OutError = TEXT("semantic PNG compression failed.");
		return false;
	}

	if (!FFileHelper::SaveArrayToFile(PngBytes, *AbsoluteOutputPath))
	{
		OutError = FString::Printf(TEXT("failed to save semantic PNG: %s"), *AbsoluteOutputPath);
		return false;
	}

	OutStats.CapturedWidth = Width;
	OutStats.CapturedHeight = Height;
	OutStats.OutputFormat = TEXT("png_semantic_rgb");
	return true;
}
