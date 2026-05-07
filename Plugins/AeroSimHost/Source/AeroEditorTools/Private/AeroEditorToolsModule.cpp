#include "AeroEditorToolsSubsystem.h"

#include "Containers/Ticker.h"
#include "Editor.h"
#include "HAL/IConsoleManager.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Modules/ModuleManager.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "PlayInEditorDataTypes.h"

namespace
{
UAeroEditorToolsSubsystem* ResolveEditorTools(FOutputDevice& Ar)
{
	if (GEditor == nullptr)
	{
		Ar.Log(TEXT("GEditor is unavailable."));
		return nullptr;
	}

	UAeroEditorToolsSubsystem* Subsystem = GEditor->GetEditorSubsystem<UAeroEditorToolsSubsystem>();
	if (Subsystem == nullptr)
	{
		Ar.Log(TEXT("AeroEditorToolsSubsystem is unavailable."));
	}

	return Subsystem;
}

void RunBootstrap(FOutputDevice& Ar)
{
	UAeroEditorToolsSubsystem* Subsystem = ResolveEditorTools(Ar);
	if (Subsystem == nullptr)
	{
		return;
	}

	FString Error;
	if (!Subsystem->BootstrapAeroWorldContentAssets(Error))
	{
		Ar.Logf(TEXT("BootstrapAeroWorldContentAssets failed: %s"), Error.IsEmpty() ? TEXT("unknown error") : *Error);
		return;
	}

	Ar.Log(TEXT("BootstrapAeroWorldContentAssets succeeded."));
}

void RunValidation(FOutputDevice& Ar)
{
	UAeroEditorToolsSubsystem* Subsystem = ResolveEditorTools(Ar);
	if (Subsystem == nullptr)
	{
		return;
	}

	FString Error;
	if (!Subsystem->ValidateAeroWorldContentAssets(Error))
	{
		Ar.Logf(TEXT("ValidateAeroWorldContentAssets failed: %s"), Error.IsEmpty() ? TEXT("unknown error") : *Error);
		return;
	}

	Ar.Log(TEXT("ValidateAeroWorldContentAssets succeeded."));
}

bool HasHeadlessBootstrapFlag()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroBootstrapWorldContent"));
}

bool HasHeadlessValidateFlag()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroValidateWorldContent"));
}

bool HasAutoPIEFlag()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroAutoPIE"));
}

FString GetCommandLineValue(const TCHAR* Key, const FString& DefaultValue)
{
	FString Value;
	if (FParse::Value(FCommandLine::Get(), Key, Value))
	{
		return Value;
	}
	return DefaultValue;
}

float GetCommandLineFloatValue(const TCHAR* Key, const float DefaultValue)
{
	float Value = DefaultValue;
	FParse::Value(FCommandLine::Get(), Key, Value);
	return Value;
}

void WriteAutoPIEStatus(const FString& Status, const FString& Detail = FString())
{
	const FString StatusFile = GetCommandLineValue(
		TEXT("AeroPIEReadyFile="),
		FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("AutoPIE"), TEXT("auto_pie_ready.json")));

	IFileManager::Get().MakeDirectory(*FPaths::GetPath(StatusFile), true);
	const FString Payload = FString::Printf(
		TEXT("{\"status\":\"%s\",\"detail\":\"%s\",\"timestamp_s\":%.3f}\n"),
		*Status.ReplaceCharWithEscapedChar(),
		*Detail.ReplaceCharWithEscapedChar(),
		FPlatformTime::Seconds());
	FFileHelper::SaveStringToFile(Payload, *StatusFile);
}

bool ShouldFailOnAutoPIETimeout()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroPIEFailOnTimeout"));
}

static FAutoConsoleCommandWithOutputDevice GAeroBootstrapWorldContentCmd(
	TEXT("aero.bootstrap_world_content"),
	TEXT("Generate or update authoritative AeroWorldContent blueprint and data assets."),
	FConsoleCommandWithOutputDeviceDelegate::CreateStatic(RunBootstrap));

static FAutoConsoleCommandWithOutputDevice GAeroValidateWorldContentCmd(
	TEXT("aero.validate_world_content"),
	TEXT("Validate authoritative AeroWorldContent assets and asset catalog mappings."),
	FConsoleCommandWithOutputDeviceDelegate::CreateStatic(RunValidation));
} // namespace

class FAeroEditorToolsModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		const bool bRunHeadlessTools = HasHeadlessBootstrapFlag() || HasHeadlessValidateFlag();
		const bool bRunAutoPIE = HasAutoPIEFlag();
		if (!bRunHeadlessTools && !bRunAutoPIE)
		{
			return;
		}

		if (bRunHeadlessTools)
		{
			FTSTicker::GetCoreTicker().AddTicker(
				FTickerDelegate::CreateLambda(
					[](float)
					{
						if (GEditor == nullptr)
						{
							return true;
						}

						UAeroEditorToolsSubsystem* Subsystem = GEditor->GetEditorSubsystem<UAeroEditorToolsSubsystem>();
						if (Subsystem == nullptr)
						{
							UE_LOG(LogTemp, Error, TEXT("Headless AeroEditorTools run failed: subsystem unavailable."));
							FPlatformMisc::RequestExit(false);
							return false;
						}

						bool bSucceeded = true;
						FString Error;
						if (HasHeadlessBootstrapFlag())
						{
							bSucceeded &= Subsystem->BootstrapAeroWorldContentAssets(Error);
							if (bSucceeded)
							{
								UE_LOG(LogTemp, Display, TEXT("Headless BootstrapAeroWorldContentAssets succeeded"));
							}
							else
							{
								UE_LOG(LogTemp, Error, TEXT("Headless BootstrapAeroWorldContentAssets failed: %s"), *Error);
							}
						}

						if (bSucceeded && HasHeadlessValidateFlag())
						{
							Error.Reset();
							bSucceeded &= Subsystem->ValidateAeroWorldContentAssets(Error);
							if (bSucceeded)
							{
								UE_LOG(LogTemp, Display, TEXT("Headless ValidateAeroWorldContentAssets succeeded"));
							}
							else
							{
								UE_LOG(LogTemp, Error, TEXT("Headless ValidateAeroWorldContentAssets failed: %s"), *Error);
							}
						}

						FPlatformMisc::RequestExit(false);
						return false;
					}),
				0.0f);
		}

		if (bRunAutoPIE)
		{
			RegisterAutoPIEHooks();
			StartAutoPIEWhenEditorIsReady();
		}
	}

	virtual void ShutdownModule() override
	{
		if (BeginPIEHandle.IsValid())
		{
			FEditorDelegates::PostPIEStarted.Remove(BeginPIEHandle);
			BeginPIEHandle.Reset();
		}
		if (EndPIEHandle.IsValid())
		{
			FEditorDelegates::EndPIE.Remove(EndPIEHandle);
			EndPIEHandle.Reset();
		}
	}

private:
	void RegisterAutoPIEHooks()
	{
		BeginPIEHandle = FEditorDelegates::PostPIEStarted.AddLambda(
			[](const bool bIsSimulating)
			{
				WriteAutoPIEStatus(TEXT("post_pie_started"), bIsSimulating ? TEXT("simulate") : TEXT("play"));
				UE_LOG(LogTemp, Display, TEXT("AeroAutoPIE: PIE fully started."));
			});

		EndPIEHandle = FEditorDelegates::EndPIE.AddLambda(
			[](const bool bIsSimulating)
			{
				WriteAutoPIEStatus(TEXT("pie_ended"), bIsSimulating ? TEXT("simulate") : TEXT("play"));
				UE_LOG(LogTemp, Warning, TEXT("AeroAutoPIE: PIE ended."));
			});
	}

	void StartAutoPIEWhenEditorIsReady()
	{
		const double StartSeconds = FPlatformTime::Seconds();
		const float StartupDelaySeconds = FMath::Max(
			0.0f,
			GetCommandLineFloatValue(TEXT("AeroPIEStartupDelaySeconds="), 8.0f));
		const float StartupTimeoutSeconds = FMath::Max(
			30.0f,
			GetCommandLineFloatValue(TEXT("AeroPIEStartupTimeoutSeconds="), 600.0f));
		const FString MapOverride = GetCommandLineValue(TEXT("AeroPIEMap="), TEXT("/Game/Maps/donghu"));
		TSharedRef<bool, ESPMode::ThreadSafe> bPlayRequested = MakeShared<bool, ESPMode::ThreadSafe>(false);

		WriteAutoPIEStatus(TEXT("waiting_for_editor"), MapOverride);
		UE_LOG(LogTemp, Display, TEXT("AeroAutoPIE: waiting for editor world, MapOverride=%s"), *MapOverride);

		FTSTicker::GetCoreTicker().AddTicker(
			FTickerDelegate::CreateLambda(
				[StartSeconds, StartupDelaySeconds, StartupTimeoutSeconds, MapOverride, bPlayRequested](float)
				{
					const double NowSeconds = FPlatformTime::Seconds();
					if (GEditor == nullptr)
					{
						return true;
					}

					if (GEditor->PlayWorld != nullptr)
					{
						WriteAutoPIEStatus(TEXT("play_world_available"), MapOverride);
						return false;
					}

					if ((NowSeconds - StartSeconds) > StartupTimeoutSeconds)
					{
						WriteAutoPIEStatus(TEXT("timeout"), TEXT("PIE did not start before AeroPIEStartupTimeoutSeconds"));
						UE_LOG(LogTemp, Error, TEXT("AeroAutoPIE: timed out before PIE started."));
						if (ShouldFailOnAutoPIETimeout())
						{
							FPlatformMisc::RequestExit(false);
						}
						return false;
					}

					if ((NowSeconds - StartSeconds) < StartupDelaySeconds)
					{
						return true;
					}

					UWorld* EditorWorld = GEditor->GetEditorWorldContext().World();
					if (EditorWorld == nullptr)
					{
						return true;
					}

					if (*bPlayRequested)
					{
						return true;
					}

					FRequestPlaySessionParams Params;
					Params.SessionDestination = EPlaySessionDestinationType::InProcess;
					Params.WorldType = EPlaySessionWorldType::PlayInEditor;
					Params.GlobalMapOverride = MapOverride;
					Params.bAllowOnlineSubsystem = false;

					*bPlayRequested = true;
					WriteAutoPIEStatus(TEXT("requesting_pie"), MapOverride);
					UE_LOG(LogTemp, Display, TEXT("AeroAutoPIE: requesting in-process PIE, MapOverride=%s"), *MapOverride);
					GEditor->RequestPlaySession(Params);
					return true;
				}),
			1.0f);
	}

	FDelegateHandle BeginPIEHandle;
	FDelegateHandle EndPIEHandle;
};

IMPLEMENT_MODULE(FAeroEditorToolsModule, AeroEditorTools)
