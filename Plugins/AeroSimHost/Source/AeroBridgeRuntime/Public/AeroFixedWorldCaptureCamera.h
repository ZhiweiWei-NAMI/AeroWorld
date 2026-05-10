#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AeroFixedWorldCaptureCamera.generated.h"

class UCameraComponent;
class USceneCaptureComponent2D;
class USceneComponent;
class UTextureRenderTarget2D;
class UMaterialInterface;
class AActor;

struct FAeroFixedWorldCaptureStats
{
	int32 CapturedWidth = 0;
	int32 CapturedHeight = 0;
	FString OutputFormat;
	bool bDepthUnitMeters = false;
	float DepthMinM = 0.0f;
	float DepthMaxM = 0.0f;
	int32 DepthValidCount = 0;
	int32 DepthInvalidCount = 0;
	FString SegmentationKind;
	FString SemanticRulesPath;
	FString SemanticAuditPath;
	TMap<uint8, FString> SemanticClassById;
	TMap<uint8, int32> SemanticClassHistogram;
	int32 IgnorePixelCount = 0;
	int32 SemanticInvalidClassIdPixelCount = 0;
	int32 SemanticUnknownColorPixelCount = 0;
	int32 SemanticAssignedComponentCount = 0;
};

UCLASS()
class AEROBRIDGERUNTIME_API AAeroFixedWorldCaptureCamera : public AActor
{
	GENERATED_BODY()

public:
	AAeroFixedWorldCaptureCamera();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	bool CaptureToDisk(
		const FString& Modality,
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		float FovDegrees,
		FString& OutError,
		FAeroFixedWorldCaptureStats& OutStats,
		const FString& SemanticRulesPath = FString(),
		const FString& SemanticAuditPath = FString());

	bool CaptureRgbToDisk(
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		float FovDegrees,
		FString& OutError,
		int32& OutCapturedWidth,
		int32& OutCapturedHeight);

private:
	bool EnsureRenderTarget(int32 Width, int32 Height, bool bFloatRenderTarget, FString& OutError);
	bool CaptureColorPngToDisk(
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		FString& OutError,
		FAeroFixedWorldCaptureStats& OutStats);
	bool CaptureDepthNpyToDisk(
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		FString& OutError,
		FAeroFixedWorldCaptureStats& OutStats);
	bool CaptureSemanticPngToDisk(
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		FString& OutError,
		FAeroFixedWorldCaptureStats& OutStats,
		const FString& SemanticRulesPath,
		const FString& SemanticAuditPath);
	void EnsureWeatherFollower();

private:
	UPROPERTY(VisibleAnywhere, Category = "Capture")
	TObjectPtr<USceneComponent> SceneRoot;

	UPROPERTY(VisibleAnywhere, Category = "Capture")
	TObjectPtr<UCameraComponent> PreviewCamera;

	UPROPERTY(VisibleAnywhere, Category = "Capture")
	TObjectPtr<USceneCaptureComponent2D> SceneCapture;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> RenderTarget;

	int32 RenderTargetWidth = 0;
	int32 RenderTargetHeight = 0;
	bool bRenderTargetFloat = false;

	UPROPERTY(Transient)
	TObjectPtr<AActor> WeatherFollowerActor;

	UPROPERTY(EditAnywhere, Category = "Weather")
	float WeatherFollowerScale = 4.0f;
};
