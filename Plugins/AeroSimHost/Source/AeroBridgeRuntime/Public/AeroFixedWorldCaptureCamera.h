#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AeroFixedWorldCaptureCamera.generated.h"

class UCameraComponent;
class USceneCaptureComponent2D;
class USceneComponent;
class UTextureRenderTarget2D;
class AActor;

UCLASS()
class AEROBRIDGERUNTIME_API AAeroFixedWorldCaptureCamera : public AActor
{
	GENERATED_BODY()

public:
	AAeroFixedWorldCaptureCamera();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	bool CaptureRgbToDisk(
		const FString& AbsoluteOutputPath,
		int32 Width,
		int32 Height,
		float FovDegrees,
		FString& OutError,
		int32& OutCapturedWidth,
		int32& OutCapturedHeight);

private:
	bool EnsureRenderTarget(int32 Width, int32 Height, FString& OutError);
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

	UPROPERTY(Transient)
	TObjectPtr<AActor> WeatherFollowerActor;

	UPROPERTY(EditAnywhere, Category = "Weather")
	float WeatherFollowerScale = 4.0f;
};
