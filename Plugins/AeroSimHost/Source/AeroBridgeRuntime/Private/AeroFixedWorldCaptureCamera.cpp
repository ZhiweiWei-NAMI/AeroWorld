#include "AeroFixedWorldCaptureCamera.h"

#include "AeroSemanticStencil.h"
#include "Annotation/ObjectAnnotator.h"
#include "Camera/CameraComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/World.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "ImageUtils.h"
#include "Materials/MaterialInterface.h"
#include "Misc/Char.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "RenderingThread.h"

#include <limits>

namespace
{
const FSoftClassPath FixedWorldCaptureWeatherActorClassPath(TEXT("AActor'/AirSim/Weather/WeatherFX/WeatherActor.WeatherActor_C'"));

void ConfigureStencilCaptureShowFlags(FEngineShowFlags& ShowFlags)
{
	FObjectAnnotator::SetViewForAnnotationRender(ShowFlags);
	ShowFlags.SetMaterials(false);
	ShowFlags.SetLighting(false);
	ShowFlags.SetBSPTriangles(true);
	ShowFlags.SetPostProcessing(true);
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
	ShowFlags.SetTextRender(false);
	ShowFlags.SetTemporalAA(false);
	ShowFlags.SetDecals(false);
}

bool ValidateCustomDepthStencilEnabled(FString& OutError)
{
	if (IConsoleVariable* CustomDepthVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.CustomDepth")))
	{
		if (CustomDepthVar->GetInt() < 3)
		{
			OutError = FString::Printf(
				TEXT("r.CustomDepth must be 3 for CustomStencil capture; current value is %d. Configure it in renderer settings/DefaultEngine.ini before capture."),
				CustomDepthVar->GetInt());
			return false;
		}
		return true;
	}
	OutError = TEXT("r.CustomDepth console variable is unavailable; CustomStencil capture cannot be validated.");
	return false;
}

uint8 DecodeSrgbByteToLinearByte(uint8 Channel)
{
	const float Srgb = static_cast<float>(Channel) / 255.0f;
	const float Linear = Srgb <= 0.04045f
		? Srgb / 12.92f
		: FMath::Pow((Srgb + 0.055f) / 1.055f, 2.4f);
	return static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(Linear * 255.0f), 0, 255));
}

bool DecodeConfiguredStencilChannel(uint8 Channel, const TSet<uint8>& AllowedClassIds, uint8& OutClassId)
{
	if (AllowedClassIds.Contains(Channel))
	{
		OutClassId = Channel;
		return true;
	}

	const uint8 LinearClassId = DecodeSrgbByteToLinearByte(Channel);
	if (AllowedClassIds.Contains(LinearClassId))
	{
		OutClassId = LinearClassId;
		return true;
	}

	return false;
}

bool DecodeConfiguredStencilPixel(const FColor& Pixel, const TSet<uint8>& AllowedClassIds, uint8& OutClassId)
{
	if (DecodeConfiguredStencilChannel(Pixel.R, AllowedClassIds, OutClassId))
	{
		return true;
	}
	if (Pixel.G != 0 && DecodeConfiguredStencilChannel(Pixel.G, AllowedClassIds, OutClassId))
	{
		return true;
	}
	if (Pixel.B != 0 && DecodeConfiguredStencilChannel(Pixel.B, AllowedClassIds, OutClassId))
	{
		return true;
	}

	OutClassId = 0;
	return false;
}

struct FSceneCaptureStateGuard
{
	explicit FSceneCaptureStateGuard(USceneCaptureComponent2D* InCapture)
		: Capture(InCapture)
		, CaptureSource(InCapture != nullptr ? InCapture->CaptureSource : ESceneCaptureSource::SCS_FinalColorLDR)
		, PrimitiveRenderMode(InCapture != nullptr ? InCapture->PrimitiveRenderMode : ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives)
		, ShowFlags(InCapture != nullptr ? InCapture->ShowFlags : FEngineShowFlags(ESFIM_Game))
		, PostProcessSettings(InCapture != nullptr ? InCapture->PostProcessSettings : FPostProcessSettings())
		, PostProcessBlendWeight(InCapture != nullptr ? InCapture->PostProcessBlendWeight : 0.0f)
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
			CapturePtr->PostProcessSettings = PostProcessSettings;
			CapturePtr->PostProcessBlendWeight = PostProcessBlendWeight;
			CapturePtr->ShowOnlyComponents = ShowOnlyComponents;
			CapturePtr->HiddenComponents = HiddenComponents;
		}
	}

	TWeakObjectPtr<USceneCaptureComponent2D> Capture;
	ESceneCaptureSource CaptureSource;
	ESceneCapturePrimitiveRenderMode PrimitiveRenderMode;
	FEngineShowFlags ShowFlags;
	FPostProcessSettings PostProcessSettings;
	float PostProcessBlendWeight;
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

bool SaveGrayscalePng(
	const FString& AbsoluteOutputPath,
	const TArray<uint8>& Pixels,
	int32 Width,
	int32 Height,
	FString& OutError)
{
	if (Pixels.Num() != Width * Height)
	{
		OutError = FString::Printf(TEXT("unexpected grayscale pixel count: expected %d got %d."), Width * Height, Pixels.Num());
		return false;
	}

	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
	const TSharedPtr<IImageWrapper> PngWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
	if (!PngWrapper.IsValid())
	{
		OutError = TEXT("failed to create PNG image wrapper.");
		return false;
	}

	if (!PngWrapper->SetRaw(Pixels.GetData(), Pixels.Num(), Width, Height, ERGBFormat::Gray, 8))
	{
		OutError = TEXT("failed to encode semantic grayscale PNG input.");
		return false;
	}

	const TArray64<uint8>& PngBytes = PngWrapper->GetCompressed(0);
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

	return true;
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
	FAeroFixedWorldCaptureStats& OutStats,
	const FString& SemanticRulesPath,
	const FString& SemanticAuditPath)
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
		RenderTarget->TargetGamma = 1.0f;
		SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
		SceneCapture->ShowFlags.SetPostProcessing(false);
		SceneCapture->ShowFlags.SetMotionBlur(false);
		SceneCapture->ShowFlags.SetDepthOfField(false);
		return CaptureDepthNpyToDisk(AbsoluteOutputPath, Width, Height, OutError, OutStats);
	}
	if (NormalizedModality.Equals(TEXT("seg"), ESearchCase::IgnoreCase))
	{
		RenderTarget->TargetGamma = 1.0f;
		SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
		ConfigureStencilCaptureShowFlags(SceneCapture->ShowFlags);
		return CaptureSemanticPngToDisk(AbsoluteOutputPath, Width, Height, OutError, OutStats, SemanticRulesPath, SemanticAuditPath);
	}

	RenderTarget->TargetGamma = 2.2f;
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
	FAeroFixedWorldCaptureStats& OutStats,
	const FString& SemanticRulesPath,
	const FString& SemanticAuditPath)
{
	if (!ValidateCustomDepthStencilEnabled(OutError))
	{
		return false;
	}

	TSet<const AActor*> IgnoredActors;
	IgnoredActors.Add(this);
	if (IsValid(WeatherFollowerActor))
	{
		IgnoredActors.Add(WeatherFollowerActor.Get());
	}

	FAeroSemanticStencilAudit Audit;
	if (!AeroSemanticStencil::AuditAndAssign(GetWorld(), SemanticRulesPath, true, IgnoredActors, Audit, OutError))
	{
		return false;
	}

	const FString CaptureEncoding = Audit.CaptureEncoding.TrimStartAndEnd().IsEmpty()
		? TEXT("custom_stencil_grayscale_u8")
		: Audit.CaptureEncoding.TrimStartAndEnd();
	if (!CaptureEncoding.Equals(TEXT("custom_stencil_grayscale_u8"), ESearchCase::IgnoreCase))
	{
		OutError = FString::Printf(TEXT("unsupported semantic stencil capture encoding '%s'."), *CaptureEncoding);
		return false;
	}
	if (Audit.CaptureMaterialPath.TrimStartAndEnd().IsEmpty())
	{
		OutError = TEXT("semantic stencil rules capture.post_process_material is required for seg capture.");
		return false;
	}
	UMaterialInterface* SemanticStencilMaterial = LoadObject<UMaterialInterface>(nullptr, *Audit.CaptureMaterialPath);
	if (!IsValid(SemanticStencilMaterial))
	{
		OutError = FString::Printf(TEXT("failed to load semantic stencil capture material: %s"), *Audit.CaptureMaterialPath);
		return false;
	}

	if (!SemanticAuditPath.TrimStartAndEnd().IsEmpty() && !AeroSemanticStencil::SaveAuditJson(Audit, SemanticAuditPath, OutError))
	{
		return false;
	}

	SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
	SceneCapture->ShowOnlyComponents.Empty();
	SceneCapture->HiddenComponents.Empty();
	SceneCapture->PostProcessSettings.WeightedBlendables.Array.Reset();
	SceneCapture->PostProcessSettings.AddBlendable(SemanticStencilMaterial, 1.0f);
	SceneCapture->PostProcessBlendWeight = 1.0f;
	FlushRenderingCommands();

	SceneCapture->CaptureScene();
	FlushRenderingCommands();

	TArray<FColor> Bitmap;
	const bool bReadOk = ReadRenderTargetColorPixels(RenderTarget, Width, Height, Bitmap, OutError);
	if (!bReadOk)
	{
		return false;
	}

	TSet<uint8> AllowedClassIds;
	for (const TPair<uint8, FString>& Pair : Audit.ClassIdToName)
	{
		AllowedClassIds.Add(Pair.Key);
	}

	TArray<uint8> ClassIdPixels;
	ClassIdPixels.SetNumUninitialized(Bitmap.Num());
	int32 InvalidClassIdPixelCount = 0;
	TMap<uint8, int32> PixelHistogram;
	for (int32 Index = 0; Index < Bitmap.Num(); ++Index)
	{
		uint8 ClassId = 0;
		if (!DecodeConfiguredStencilPixel(Bitmap[Index], AllowedClassIds, ClassId))
		{
			++InvalidClassIdPixelCount;
		}
		ClassIdPixels[Index] = ClassId;
		PixelHistogram.FindOrAdd(ClassId) += 1;
	}

	if (!SaveGrayscalePng(AbsoluteOutputPath, ClassIdPixels, Width, Height, OutError))
	{
		return false;
	}

	OutStats.CapturedWidth = Width;
	OutStats.CapturedHeight = Height;
	OutStats.OutputFormat = TEXT("png_uint8_class_id");
	OutStats.SegmentationKind = TEXT("ue_custom_stencil_class_id_u8");
	OutStats.SemanticRulesPath = Audit.RulesPath;
	OutStats.SemanticAuditPath = SemanticAuditPath;
	OutStats.SemanticClassById = Audit.ClassIdToName;
	OutStats.SemanticClassHistogram = PixelHistogram;
	OutStats.IgnorePixelCount = PixelHistogram.FindRef(0);
	OutStats.SemanticInvalidClassIdPixelCount = InvalidClassIdPixelCount;
	OutStats.SemanticUnknownColorPixelCount = InvalidClassIdPixelCount;
	OutStats.SemanticAssignedComponentCount = Audit.AssignedComponentCount;
	return true;
}
